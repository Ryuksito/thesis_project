import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONSTANTES DE DIMENSIONALIDAD
# ==========================================
MAX_ATOMS = 24           # ¡El nuevo límite dorado!
LATTICE_PARAMS = 6       # a, b, c, alpha, beta, gamma
MAX_ELEMENTS = 5
ELEM_FEATURES = 9
CRYSTAL_EMBED = 32
CRYSTAL_PROPS = 3

INPUT_DIM  = (MAX_ELEMENTS * ELEM_FEATURES) + CRYSTAL_EMBED + CRYSTAL_PROPS # 112

# 🔴 CORRECCIÓN CRÍTICA:
# NEAT escupe el VECTOR LATENTE (64) que luego será devorado por el Decoder.
OUTPUT_DIM = 64  

# ==========================================
# CONSTANTES DE NORMALIZACIÓN
# ==========================================
MAX_ATOMIC_NUMBER = 118.0 
MAX_LATTICE_LENGTH = 38.0 
MAX_LATTICE_ANGLE = 180.0

# ==========================================
# CONSTANTES DE ENTRENAMIENTO JAX/NEAT
# ==========================================

# DATA_PATH = os.getenv("DATA_PATH")
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "_dataset")
SEED = 42

# --- Hiperparámetros de Descenso de Gradiente ---
# 🟢 Subido de 8 a 128. JAX necesita batches grandes para amortizar la compilación.
BATCH_SIZE = 32            
N_GENERATIONS = 1500
GRAD_STEPS_PER_GEN = 10
TOTAL_GRAD_STEPS = N_GENERATIONS * GRAD_STEPS_PER_GEN # 5000 iteraciones totales

LR_MAX = 0.005              # Tasa de aprendizaje máxima (arranque rápido)
LR_MIN = 1e-5               # Tasa mínima (aterrizaje suave en el mínimo global)

# --- Hiperparámetros Topológicos (TensorNEAT) ---
# 🟢 Subido de 112 a 1000. Tendrás 1000 topologías neuronales distintas compitiendo.
POPSIZE = 64     
POP_BATCH_SIZE = 16       
SPECIES_SIZE = 20
SURVIVAL_THRESHOLD = 0.1

# 🟢 Reducidos para ahorrar VRAM y obligar a NEAT a ser elegante y directo.
MAX_NODES = 400
MAX_CONNS = 5000
CONN_ADD_PROB = 0.15
CONN_DELETE_PROB = 0.3
NODE_ADD_PROB = 0.05
NODE_DELETE_PROB = 0.05