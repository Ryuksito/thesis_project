import h5py
import numpy as np
from dataclasses import dataclass

# =================================================================
# 1. DATACLASSES
# Utilizamos una clase base única para evitar repetir código, 
# ya que Train y Test tienen exactamente la misma estructura.
# =================================================================
@dataclass
class DatasetSplit:
    context_elements: np.ndarray
    context_embeddings: np.ndarray
    context_props: np.ndarray
    target_lattice: np.ndarray
    target_atoms: np.ndarray
    target_positions: np.ndarray
    material_ids: np.ndarray

    def __len__(self):
        return len(self.material_ids)

# =================================================================
# 2. CARGA DESDE HDF5 A LA MEMORIA RAM
# =================================================================
def load_dataset(h5_path: str) -> tuple[DatasetSplit, DatasetSplit]:
    """
    Carga todo el HDF5 en la memoria RAM (solo pesa unos pocos MBs).
    Retorna la partición de Train y la de Test listas para usarse.
    """
    print(f"📂 Cargando dataset en memoria desde: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        def extract_split(group_name: str) -> DatasetSplit:
            grp = hf[group_name]
            return DatasetSplit(
                context_elements=grp["context_elements"][:],
                context_embeddings=grp["context_embeddings"][:],
                context_props=grp["context_props"][:],
                target_lattice=grp["target_lattice"][:],
                target_atoms=grp["target_atoms"][:],
                target_positions=grp["target_positions"][:],
                # Decodificamos los strings de bytes a strings normales de Python
                material_ids=np.array([mid.decode('utf-8') if isinstance(mid, bytes) else str(mid) for mid in grp["material_ids"][:]])
            )
            
        train_ds = extract_split("train")
        test_ds = extract_split("test")
        
    print(f"✅ Cargado exitosamente: Train ({len(train_ds)}), Test ({len(test_ds)})")
    return train_ds, test_ds

# =================================================================
# 3. CARGADOR CIRCULAR INTELIGENTE (Para JAX/NEAT)
# =================================================================
class BatchLoader:
    def __init__(self, dataset: DatasetSplit, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(dataset)
        
        # Índices para llevar el control
        self.indices = np.arange(self.n)
        self.cursor = 0
        self.epochs_completed = 0

        # Barajar por primera vez si es necesario
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """
        Generador infinito que SIEMPRE devuelve un diccionario 
        con matrices del tamaño exacto de `batch_size`.
        """
        batch_indices = []

        # Bucle envolvente (Wrap-around)
        while len(batch_indices) < self.batch_size:
            necesitamos = self.batch_size - len(batch_indices)
            disponibles_en_epoch = self.n - self.cursor

            if disponibles_en_epoch > necesitamos:
                # Caso Normal: Tenemos suficientes datos en esta vuelta
                batch_indices.extend(self.indices[self.cursor : self.cursor + necesitamos])
                self.cursor += necesitamos
            else:
                # Caso Borde: Se nos acaban los datos de esta vuelta.
                # Tomamos lo que queda...
                batch_indices.extend(self.indices[self.cursor : self.n])
                
                # ...y reiniciamos el ciclo (Nueva Época)
                self.cursor = 0
                self.epochs_completed += 1
                if self.shuffle:
                    np.random.shuffle(self.indices)

        # Convertir la lista de índices a un array de numpy
        idx_array = np.array(batch_indices)

        # Retornamos el batch como un diccionario fácil de desempaquetar
        return {
            "context_elements": self.dataset.context_elements[idx_array],
            "context_embeddings": self.dataset.context_embeddings[idx_array],
            "context_props": self.dataset.context_props[idx_array],
            "target_lattice": self.dataset.target_lattice[idx_array],
            "target_atoms": self.dataset.target_atoms[idx_array],
            "target_positions": self.dataset.target_positions[idx_array],
            "material_ids": self.dataset.material_ids[idx_array]
        }