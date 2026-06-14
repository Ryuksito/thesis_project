import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONSTANTES DE DIMENSIONALIDAD
# ==========================================
MAX_ATOMS = 24           # ¡El nuevo límite dorado!
LATTICE_PARAMS = 6       # a, b, c, alpha, beta, gamma

# ==========================================
# CONSTANTES DE NORMALIZACIÓN
# ==========================================
MAX_ATOMIC_NUMBER = 118
MAX_LATTICE_LENGTH = 38.0  # Ajusta esto si en tu anterior código usabas otro valor
MAX_LATTICE_ANGLE = 180.0

# ==========================================
# STRUC CONSTANTS
# ==========================================
DATA_PATH = os.getenv("DATA_PATH")
BATCH_SIZE = 256
MAX_N_EDGES = 131072
SEED = 42
EPOCHS = 5