import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from matscipy.neighbours import neighbour_list

def lattice_params_to_matrix(params):
    """Convierte [a, b, c, alpha, beta, gamma] en matriz cartesiana 3x3."""
    params = np.asarray(params, dtype=np.float64)
    a, b, c, alpha, beta, gamma = params
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * np.cos(gamma_r), b * np.sin(gamma_r), 0.0], dtype=np.float64)
    
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))
    vc = np.array([cx, cy, cz], dtype=np.float64)
    
    return np.vstack([va, vb, vc])

class CrystalDataset(Dataset):
    """
    Dataset de PyTorch que carga los datos HDF5 en RAM y construye
    los grafos de PyTorch Geometric al vuelo (en la CPU).
    """
    def __init__(self, h5_path, split='train', max_lattice_length=1.0, 
                 max_lattice_angle=1.0, cutoff=4.0):
        super().__init__()
        self.max_lattice_length = max_lattice_length
        self.max_lattice_angle = max_lattice_angle
        self.cutoff = cutoff

        print(f"📥 Cargando datos de la partición '{split}' en memoria...")
        with h5py.File(h5_path, 'r') as hf:
            grp = hf[split]
            self.target_lattice = grp['target_lattice'][:]
            self.target_atoms = grp['target_atoms'][:]
            self.target_positions = grp['target_positions'][:]
            self.material_ids = grp['material_ids'][:]
            
        print(f"✅ ¡{len(self.material_ids)} cristales cargados desde '/{split}'!")

    def __len__(self):
        return len(self.material_ids)

    def __getitem__(self, idx):
        # 1. Extraer los datos ya cargados en RAM (NORMALIZADOS [0,1])
        norm_lattice = np.asarray(self.target_lattice[idx], dtype=np.float64)
        Z_val = np.asarray(self.target_atoms[idx], dtype=np.float64)
        frac_positions = np.asarray(self.target_positions[idx], dtype=np.float64)
        
        # 2. Crear copia desnormalizada explícita para la física
        real_lattice = norm_lattice.copy()
        real_lattice[0:3] *= self.max_lattice_length
        real_lattice[3:6] *= self.max_lattice_angle
        
        # 3. Extraer átomos reales (padding Z=0 heredado del guardado)
        real_mask = Z_val > 0
        Z_numbers = Z_val[real_mask]
        real_frac_pos = frac_positions[real_mask]
        
        num_nodes = len(Z_numbers)
        
        # Validación: Si el grafo es inválido, sacamos otro al azar
        if num_nodes == 0: 
            return self.__getitem__(np.random.randint(len(self)))

        # ¡OJO! Usamos el real_lattice para calcular distancias reales
        cell_matrix = lattice_params_to_matrix(real_lattice)
        volume = np.abs(np.linalg.det(cell_matrix))
        if volume < 1e-4:
            return self.__getitem__(np.random.randint(len(self)))

        cartesian_positions = real_frac_pos @ cell_matrix

        # Calcular aristas con PBC
        try:
            receivers, senders, senders_unit_shifts = neighbour_list(
                quantities="ijS",
                pbc=np.array([True, True, True]),
                cell=cell_matrix,
                positions=cartesian_positions,
                cutoff=self.cutoff,
            )
        except Exception:
            return self.__getitem__(np.random.randint(len(self)))

        # Formatear a tensores de PyTorch
        edge_index = torch.tensor(np.vstack([senders, receivers]), dtype=torch.long)
        
        # PyG Data object
        data = Data(
            x=torch.tensor(Z_numbers, dtype=torch.long),
            pos=torch.tensor(real_frac_pos, dtype=torch.float32),
            edge_index=edge_index,
            edge_shift=torch.tensor(senders_unit_shifts, dtype=torch.float32), # <-- ¡CRÍTICO PARA PBC!
            lattice=torch.tensor(real_lattice, dtype=torch.float32).unsqueeze(0), # Para la física del Encoder
            norm_lattice=torch.tensor(norm_lattice, dtype=torch.float32).unsqueeze(0), # Para el Loss
            id=str(self.material_ids[idx])
        )

        return data