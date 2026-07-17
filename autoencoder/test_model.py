# ==========================================
# CONFIGURACIÓN DE RUTAS (Root Execution)
# ==========================================
import os
import sys

if os.getcwd().endswith("autoencoder"):
    os.chdir("../")
sys.path.append(os.getcwd())    
print(f"Current working directory: {os.getcwd()}")
print(f"Python import paths: {sys.path[-1]}")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np 

from autoencoder.models.autoencoder import Autoencoder, CrystalDecoder
from autoencoder.config import *
from autoencoder.data.loader import load_dataset, get_batches, create_batched_dataset
from autoencoder.loss import mse_loss


# LOGS_DIR = "/home/alanh/Dev/owns/thesis_project/autoencoder/runs/v1/"
BASE_DIR = os.getcwd()
LOGS_DIR = os.path.join(BASE_DIR, "autoencoder", "runs", "v1")
print(f"Los logs se guardarán en: {LOGS_DIR}")
CHECKPOINT_DIR = os.path.join(LOGS_DIR, "checkpoints")

# =====================================================================
# 🧬 FILTRO GEOMÉTRICO INVARIANTE (NEAT SURVIVAL LOSS) - VERSIÓN SEPARADA
# =====================================================================
@jax.jit
def crystal_loss_fn(pred_lat, pred_pos, pred_z, target_lat, target_pos, target_z):
    """
    Pérdida para la Supervivencia NEAT. 
    Recibe Lattice, Posiciones y Átomos (Z) ya separados.
    """
    # Máscaras de átomos reales
    mask = (target_z > 0).astype(jnp.float32)
    mask_2d = mask[:, :, None] * mask[:, None, :]
    
    # 1. LATTICE
    lat_loss = jnp.mean((pred_lat - target_lat)**2)

    # 2. ACSF (Atom-Centered Symmetry Functions con PBC)
    def get_sorted_fingerprints(pos, m2d, eta=5.0):
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        diff = diff - jnp.round(diff) # Condiciones periódicas de contorno (PBC)
        dist_sq = jnp.sum(diff**2, axis=-1)
        gaussians = jnp.exp(-eta * dist_sq) * m2d
        
        eye = jnp.eye(pos.shape[1])[None, :, :]
        gaussians = gaussians * (1.0 - eye)
        
        fingerprints = jnp.sum(gaussians, axis=-1)
        return jnp.sort(fingerprints, axis=-1)

    # Ya no rebanamos nada, le pasamos las posiciones puras XYZ (dim 3)
    fp_pred = get_sorted_fingerprints(pred_pos, mask_2d)
    fp_target = get_sorted_fingerprints(target_pos, mask_2d)
    acsf_loss = jnp.sum(((fp_pred - fp_target)**2) * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # 3. Z-LOSS INVARIANTE
    sorted_pred_z = jnp.sort(pred_z * mask, axis=1)
    sorted_target_z = jnp.sort(target_z, axis=1)
    z_loss = jnp.mean((sorted_pred_z - sorted_target_z)**2)

    # 4. VARIANCE & REPULSION
    fp_var_pred = jnp.var(fp_pred * mask, axis=0)
    fp_var_target = jnp.var(fp_target * mask, axis=0)
    var_loss = jnp.mean((fp_var_pred - fp_var_target)**2)
    
    repulsion = jnp.exp(-1000.0 * (fp_pred + 1e-6)) * mask
    repulsion_loss = jnp.sum(repulsion) / jnp.maximum(jnp.sum(mask), 1.0)

    # Pesos del filtro evolutivo
    w_lat = 10.0 * lat_loss
    w_acsf = 50.0 * acsf_loss
    w_z = 20.0 * z_loss
    w_var = 10.0 * var_loss
    w_rep = 10.0 * repulsion_loss

    total_loss = w_lat + w_acsf + w_z + w_var + w_rep
    
    # Retornamos siempre la tupla completa
    return total_loss, w_lat, w_acsf, w_z, w_var, w_rep


def load_and_test():
    print("\n" + "="*60)
    print("🧪 INICIANDO PRUEBA DE INFERENCIA ESTRICTA (MSE EVALUATION)")
    print("="*60)

    # 1. Definir el modelo completo
    model = Autoencoder(latent_dim=64, max_atoms=MAX_ATOMS)
    
    # 2. Cargar datos
    print("Cargando datos...")
    dset = load_dataset(DATA_PATH + "/autoencoder_16k.h5")
    dummy_batch = list(get_batches(dset, BATCH_SIZE, shuffle=False))[0] 
    
    dummy_graph = create_batched_dataset(
        dummy_batch['atoms'], dummy_batch['positions'], dummy_batch['lattice'],
        BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
    )
    
    # 3. Manager de Checkpoints
    checkpoint_manager = ocp.CheckpointManager(os.path.abspath(CHECKPOINT_DIR), ocp.PyTreeCheckpointer())
    best_step = checkpoint_manager.best_step()
    if best_step is None:
        print("❌ No se encontró ningún checkpoint.")
        return
        
    print(f"Cargando checkpoint del paso: {best_step}...\n")
    restored = checkpoint_manager.restore(best_step, args=ocp.args.PyTreeRestore())
    params = restored['params'] if 'params' in restored else restored

    # --- Extracción y Cirugía ---
    real_latent_vector = model.apply({'params': params}, dummy_graph, method=model.encode)
    decoder_only_params = params['decoder']
    
    del params, restored, model

    # 5. INFERENCIA PURA
    np.set_printoptions(precision=4, suppress=True)
    decoder_model = CrystalDecoder(max_atoms=MAX_ATOMS)
    
    pred_lattice_raw, pred_pos_raw, pred_z_raw = decoder_model.apply(
        {'params': decoder_only_params}, 
        real_latent_vector
    )
    
    # Recorte del padding de Jraph
    pred_lattice_full = pred_lattice_raw[:-1]
    pred_pos_full = pred_pos_raw[:-1]
    pred_z_full = pred_z_raw[:-1]
    
    target_lattice = jnp.array(dummy_batch['lattice'])
    target_pos = jnp.array(dummy_batch['positions'])
    target_z = jnp.array(dummy_batch['atoms'])

    # =========================================================================
    # 🧬 EVALUACIÓN DIRECTA DEL FILTRO NEAT (Sin necesidad de concatenar)
    # =========================================================================
    neat_total, n_lat, n_acsf, n_z, n_var, n_rep = crystal_loss_fn(
        pred_lattice_full,  
        pred_pos_full, 
        pred_z_full, 
        target_lattice, 
        target_pos, 
        target_z
    )

    total_loss, (loss_lattice, loss_z, loss_pos) = mse_loss(
        pred_lattice_full, 
        pred_pos_full, 
        pred_z_full, 
        target_lattice, 
        target_pos, 
        target_z
    )

    # --- IMPRESIÓN DE RESULTADOS ---
    print("="*60)
    print("🧬 EVALUACIÓN DEL FILTRO EVOLUTIVO (NEAT SURVIVAL LOSS)")
    print("="*60)
    print(f" ├─ Score Total (Fitness Penalty): {neat_total:.4f}")
    print(f" ├─ Penalización Lattice (w=10):   {n_lat:.4f}")
    print(f" ├─ Penalización ACSF (w=50):      {n_acsf:.4f}  <-- Error Estructural 3D")
    print(f" ├─ Penalización Z-Inv (w=20):     {n_z:.4f}  <-- Error Químico")
    print(f" ├─ Penalización Varianza (w=10):  {n_var:.4f}")
    print(f" └─ Penalización Repulsión (w=10): {n_rep:.4f}\n")
    print("="*60)
    print("🧬 EVALUACIÓN DEL LOSS")
    print("="*60)
    print(f" ├─ Score Total: {total_loss:.4f}")
    print(f" ├─ Penalización loss_lattice:   {loss_lattice:.4f}")
    print(f" ├─ Penalización loss_z:      {loss_z:.4f}")
    print(f" └─ Penalización loss_pos:     {loss_pos:.4f}\n")
    
    # Buscar el primer cristal válido
    idx = 0
    while idx < len(target_z) and np.sum(target_z[idx]) == 0:
        idx += 1
        
    if idx < len(target_z):
        real_z = target_z[idx]
        pred_z = pred_z_full[idx]
        num_atoms = int(np.sum(real_z > 0))
        real_z_tabla = np.round(real_z * MAX_ATOMIC_NUMBER)
        pred_z_tabla = np.round(pred_z * MAX_ATOMIC_NUMBER)
        
        print(f"🔬 INSPECCIÓN VISUAL (Cristal #{idx})")
        print(f"⚛️  IDENTIDADES ATÓMICAS (Z)")
        for i in range(num_atoms):
            print(f"[{i}]: real = {real_z_tabla[i]} | pred = {pred_z_tabla[i]}")

if __name__ == "__main__":
    load_and_test()