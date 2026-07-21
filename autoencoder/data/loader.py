# autoencoder/data/loader.py
import h5py
import numpy as np
import jraph
from matscipy.neighbours import neighbour_list
from dataclasses import dataclass

@dataclass
class DatasetBase:
    lattice: np.ndarray
    atoms: np.ndarray
    positions: np.ndarray
    ids: np.ndarray

# =====================================================================
# 1. CARGA DE DATOS A LA MEMORIA RAM
def load_dataset(h5_path, split='train') -> DatasetBase:
    """
    Carga TODO el dataset HDF5 en la memoria RAM para evitar la inanición 
    de la GPU (GPU Starvation) durante el entrenamiento asíncrono con JAX.
    
    Args:
        h5_path (str): Ruta al archivo .h5
        split (str): Partición a cargar ('train' o 'test'). Por defecto es 'train'.
    """    
    print(f"📥 Cargando datos de la partición '{split}' en memoria...")
    
    with h5py.File(h5_path, 'r') as hf:
        # 1. Apuntamos específicamente al subgrupo ('train' o 'test')
        grp = hf[split]
        
        # 2. Extraemos los tensores desde ese subgrupo
        target_lattice = grp['target_lattice'][:]
        target_atoms = grp['target_atoms'][:]
        target_positions = grp['target_positions'][:]
        material_ids = grp['material_ids'][:]
        
    print(f"✅ ¡{len(material_ids)} cristales cargados desde '/{split}'!")
    
    return DatasetBase(
        lattice=target_lattice,
        atoms=target_atoms,
        positions=target_positions,
        ids=material_ids
    )

def get_batches(dataset: DatasetBase, batch_size, shuffle=True):
    """Generador que entrega mini-batches directamente desde la RAM."""
    num_samples = len(dataset.ids)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        
    num_batches = num_samples // batch_size
    
    for i in range(num_batches):
        batch_idx = indices[i * batch_size : (i + 1) * batch_size]
        yield {
            "lattice": dataset.lattice[batch_idx],
            "atoms": dataset.atoms[batch_idx],
            "positions": dataset.positions[batch_idx],
            "ids": dataset.ids[batch_idx]
        }

# =====================================================================
# 2. TRANSFORMACIÓN GEOMÉTRICA (Cartesianas y PBC)
# =====================================================================
def lattice_params_to_matrix(params):
    """Convierte [a, b, c, alpha, beta, gamma] desnormalizado en matriz cartesiana 3x3."""
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

def matrix_to_graph(Z_array, frac_positions, lattice_params, 
                    max_atomic_number, max_lattice_length, max_lattice_angle, 
                    cutoff=4.0):
    """Convierte arrays individuales en un jraph.GraphsTuple inyectando las constantes."""
    
    Z_val = np.asarray(Z_array, dtype=np.float64)
    
    frac_positions = np.asarray(frac_positions, dtype=np.float64)
    
    lattice_params = np.asarray(lattice_params, dtype=np.float64)
    lattice_params[0:3] *= max_lattice_length
    lattice_params[3:6] *= max_lattice_angle
    
    # 2. Extraer átomos reales (padding Z=0)
    real_mask = Z_val > 0
    Z_numbers = Z_val[real_mask]
    real_frac_pos = frac_positions[real_mask]
    
    num_nodes = len(Z_numbers)
    if num_nodes == 0: 
        return None

    # 3. Construir la matriz de la celda unitaria con los datos reales
    cell_matrix = lattice_params_to_matrix(lattice_params)

    volume = np.abs(np.linalg.det(cell_matrix))
    if volume < 1e-4:
        # Silenciamos el print para no spamear durante el entrenamiento real
        return None

    # 4. Cartesianas absolutas para matscipy
    cartesian_positions = real_frac_pos @ cell_matrix

    # 5. Calculamos las aristas con PBC (matscipy)
    try:
        receivers, senders, senders_unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=np.array([True, True, True]),
            cell=cell_matrix,
            positions=cartesian_positions,
            cutoff=cutoff,
        )
    except Exception:
        return None
    
    num_edges = senders.shape[0]

    # 6. Creamos el GraphsTuple
    return jraph.GraphsTuple(
        nodes={"Z": Z_numbers, "pos": real_frac_pos}, 
        edges={"shifts": senders_unit_shifts},
        globals={"lattice": lattice_params[None, :]}, 
        senders=senders,
        receivers=receivers,
        n_node=np.array([num_nodes]),
        n_edge=np.array([num_edges]),
    )

# =====================================================================
# 3. ENSAMBLADOR DE BATCHES ESTÁTICOS PARA JAX
# =====================================================================
def create_batched_dataset(Z_batch, positions_batch, lattice_batch, 
                           batch_size, max_atoms, max_n_edges,
                           max_atomic_number, max_lattice_length, max_lattice_angle,
                           cutoff=4.0):
    """
    Toma un mini-batch y lo empaqueta en un GraphsTuple rígido con padding,
    usando los parámetros inyectados desde config.py.
    """
    graphs = []
    for i in range(len(Z_batch)):
        g = matrix_to_graph(
            Z_batch[i], positions_batch[i], lattice_batch[i],
            max_atomic_number, max_lattice_length, max_lattice_angle,
            cutoff
        )
        if g is not None:
            graphs.append(g)
            
    if not graphs:
        # En vez de romper todo con ValueError, podemos devolver un grafo "falso" 
        # para que JAX no se caiga, pero por ahora lo dejamos como error estricto
        raise ValueError("Todos los grafos del batch estaban vacíos o colapsados.")
            
    batched_graph = jraph.batch(graphs)
    
    # LA CLAVE DE JAX: LÍMITES ESTÁTICOS Y ABSOLUTOS
    max_n_node = (max_atoms * batch_size) + 1 
    max_n_graph = batch_size + 1
    
    padded_graph = jraph.pad_with_graphs(
        batched_graph, 
        n_node=max_n_node, 
        n_edge=max_n_edges, 
        n_graph=max_n_graph
    )
    
    return padded_graph