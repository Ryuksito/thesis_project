# ==========================================
# CONFIGURACIÓN DE RUTAS (Root Execution)
# ==========================================
import os
import sys
import time
import json
import subprocess

if os.getcwd().endswith("autoencoder"):
    os.chdir("../")
sys.path.append(os.getcwd())    
print(f"Current working directory: {os.getcwd()}")
print(f"Python import paths: {sys.path[-1]}")

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
import numpy as np
from tqdm import tqdm

# Tus módulos
from autoencoder.config import *
from autoencoder.data.loader import load_dataset, get_batches, create_batched_dataset
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.loss import compute_total_loss

LOGS_DIR = "/home/alanh/Dev/owns/thesis_project/autoencoder/runs/v2/"

# =====================================================================
# FUNCIÓN AUXILIAR PARA OBTENER VRAM
# =====================================================================
def get_gpu_vram():
    """Obtiene el uso actual de VRAM en MB usando nvidia-smi."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        vram_used = result.strip().split('\n')[0]
        return f"{vram_used} MB"
    except Exception:
        return "N/A"

# =====================================================================
# PASO DE ENTRENAMIENTO COMPILADO (JIT)
# =====================================================================
@jax.jit
def train_step(state, graph, target_lattice, target_pos, target_z):
    
    def loss_fn(params):
        # 1. El modelo genera las predicciones (Incluyendo el grafo de padding)
        pred_lattice_raw, pred_pos_raw, pred_z_logits_raw, _ = state.apply_fn({'params': params}, graph)
        
        # 2. 👻 ELIMINAMOS EL GRAFO FANTASMA DE JRAPH ANTES DEL LOSS
        pred_lattice = pred_lattice_raw[:-1]
        pred_pos = pred_pos_raw[:-1]
        pred_z_logits = pred_z_logits_raw[:-1]
        
        # 3. Calculamos la pérdida usando las funciones matemáticas puras (MSE, CE, ACSF)
        loss, aux = compute_total_loss(
            pred_lattice, pred_pos, pred_z_logits, 
            target_lattice, target_pos, target_z
        )
        return loss, aux

    # Calculamos gradientes (Backward pass)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(state.params)
    
    # Optimizamos los pesos
    state = state.apply_gradients(grads=grads)
    return state, loss, aux_metrics

# =====================================================================
# BUCLE PRINCIPAL
# =====================================================================
def main():
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO DEL AUTOENCODER CRISTALOGRÁFICO")
    print("="*60)
    
    # --- PREPARACIÓN DE DIRECTORIOS ---
    os.makedirs(LOGS_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(LOGS_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    json_log_path = os.path.join(LOGS_DIR, "training_logs.json")
    all_logs = {}
    
    # Manager de Checkpoints (Orbax)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1, 
        create=True,
        best_fn=lambda metric: metric, 
        best_mode='min'
    )
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(CHECKPOINT_DIR), 
        ocp.PyTreeCheckpointer(), 
        options=options
    )
    
    # 1. Cargar Datos
    dset = load_dataset(DATA_PATH + "/autoencoder_16k.h5")
    total_batches = len(dset.ids) // BATCH_SIZE
    
    # 2. Inicializar Modelo
    key = jax.random.PRNGKey(SEED)
    model = Autoencoder(latent_dim=64, max_atoms=MAX_ATOMS)
    
    print("\n🧠 Compilando modelo y asignando pesos iniciales...")
    dummy_batch = list(get_batches(dset, BATCH_SIZE, shuffle=False))[0]
    dummy_graph = create_batched_dataset(
        dummy_batch['atoms'], dummy_batch['positions'], dummy_batch['lattice'],
        BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
    )
    variables = model.init(key, dummy_graph)
    
    # 3. Optimizador y Estado
    optimizer = optax.adamw(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model, params=variables, tx=optimizer)
    
    # 4. Restaurar Checkpoint si existe (Para no perder progreso)
    best_step = checkpoint_manager.best_step()
    start_epoch = 0
    if best_step is not None:
        print(f"\n🔄 ¡Checkpoint encontrado! Restaurando pesos de la época {best_step}...")
        restored_state = checkpoint_manager.restore(best_step, args=ocp.args.PyTreeRestore())
        # Actualizamos solo los parámetros, mantenemos la estructura de TrainState
        state = state.replace(params=restored_state['params'] if 'params' in restored_state else restored_state)
        start_epoch = best_step
        
        # Recuperamos el historial de logs para no sobreescribirlo
        if os.path.exists(json_log_path):
            with open(json_log_path, 'r') as f:
                all_logs = json.load(f)
            print("📊 Historial JSON anterior cargado correctamente.")

    # 5. Entrenamiento
    print(f"\n🔥 ¡Arrancando GPU! Entrenando hasta la época {EPOCHS}...\n")
    
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        batch_losses, batch_lat_losses, batch_pos_losses, batch_z_losses = [], [], [], []
        
        # Barra de progreso TQDM
        pbar = tqdm(get_batches(dset, BATCH_SIZE, shuffle=True), total=total_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            try:
                graph = create_batched_dataset(
                    batch['atoms'], batch['positions'], batch['lattice'],
                    BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
                )
            except ValueError:
                continue # Saltamos batches vacíos
            
            # ATENCIÓN AQUÍ: Los target_z ya no se normalizan, deben ser enteros puros crudos del dataset
            target_lattice = jnp.array(batch['lattice'])
            target_pos = jnp.array(batch['positions'])
            target_z = jnp.array(batch['atoms'])
            
            # Forward + Backward
            state, loss, (l_lat, l_z, l_pos) = train_step(state, graph, target_lattice, target_pos, target_z)
            
            # Guardar para promedios
            batch_losses.append(loss.item())
            batch_lat_losses.append(l_lat.item())
            batch_z_losses.append(l_z.item())
            batch_pos_losses.append(l_pos.item())
            
            # Actualizar barra visual
            pbar.set_postfix({'Loss': f"{np.mean(batch_losses[-50:]):.4f}"})
                
        end_time = time.time()
        epoch_time = end_time - start_time
        avg_loss = float(np.mean(batch_losses))
        
        # Logging dinámico
        all_logs[str(epoch + 1)] = {
            "loss": batch_losses,
            "lattice_loss": batch_lat_losses,
            "position_loss": batch_pos_losses,
            "z_loss": batch_z_losses,
            "vram": get_gpu_vram(),
            "time": float(epoch_time)
        }
        
        with open(json_log_path, 'w') as f:
            json.dump(all_logs, f, indent=4)

        # 6. Guardar Checkpoint
        checkpoint_manager.save(
            epoch + 1, # Guardamos como epoch+1 para que Orbax entienda la secuencia
            args=ocp.args.PyTreeSave(state.params), 
            metrics=avg_loss
        )
        
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Guardado JSON actualizado.")

    checkpoint_manager.wait_until_finished()
    print("\n🎉 ENTRENAMIENTO FINALIZADO. Pesos guardados con éxito.")

if __name__ == "__main__":
    main()