# ==========================================
# CONFIGURACIÓN DE RUTAS (Root Execution)
# ==========================================
import os
import sys
import time
import subprocess
import csv  # <--- IMPORTANTE: Agregado para manejar el CSV

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

# LOGS_DIR = "/home/alanh/Dev/owns/thesis_project/autoencoder/runs/v2/"

BASE_DIR = os.getcwd()
LOGS_DIR = os.path.join(BASE_DIR, "autoencoder", "runs", "v2")
print(f"Los logs se guardarán en: {LOGS_DIR}")

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
        pred_lattice_raw, pred_pos_raw, pred_z_logits_raw, _ = state.apply_fn({'params': params}, graph)
        pred_lattice = pred_lattice_raw[:-1]
        pred_pos = pred_pos_raw[:-1]
        pred_z_logits = pred_z_logits_raw[:-1]
        loss, aux = compute_total_loss(
            pred_lattice, pred_pos, pred_z_logits, 
            target_lattice, target_pos, target_z
        )
        return loss, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_metrics), grads = grad_fn(state.params)
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
    
    # 📝 Cambiamos a CSV
    csv_log_path = os.path.join(LOGS_DIR, "training_logs.csv")
    
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
    
    dset = load_dataset(DATA_PATH + "/autoencoder_16k.h5")
    total_batches = len(dset.ids) // BATCH_SIZE
    
    key = jax.random.PRNGKey(SEED)
    model = Autoencoder(latent_dim=64, max_atoms=MAX_ATOMS)
    
    print("\n🧠 Compilando modelo y asignando pesos iniciales...")
    dummy_batch = list(get_batches(dset, BATCH_SIZE, shuffle=False))[0]
    dummy_graph = create_batched_dataset(
        dummy_batch['atoms'], dummy_batch['positions'], dummy_batch['lattice'],
        BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
    )
    variables = model.init(key, dummy_graph)
    
    optimizer = optax.adamw(learning_rate=1e-5)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer)
    
    # 4. Restaurar Checkpoint si existe
    best_step = checkpoint_manager.best_step()
    start_epoch = 0
    if best_step is not None:
        print(f"\n🔄 ¡Checkpoint encontrado! Analizando formato de la época {best_step}...")
        
        # Leemos el archivo en bruto primero para ver qué tiene adentro
        raw_restored = checkpoint_manager.restore(best_step)
        
        if 'opt_state' in raw_restored:
            # FORMATO NUEVO: Tiene la memoria del optimizador
            print("📦 Formato NUEVO detectado. Restaurando pesos + inercia de AdamW...")
            state = checkpoint_manager.restore(best_step, args=ocp.args.PyTreeRestore(item=state))
        else:
            # FORMATO VIEJO: Solo tiene los pesos (Época 37 o anteriores)
            print("⚠️ Formato ANTIGUO detectado (solo pesos).")
            print("   -> Ocurrirá un ligero 'Optimizer Shock' en la primera época, es normal.")
            params_only = raw_restored['params'] if 'params' in raw_restored else raw_restored
            state = state.replace(params=params_only)
            
        start_epoch = best_step
        print("📊 El historial CSV continuará escribiéndose sin borrar lo anterior.")
    else:
        # Si empezamos de cero, creamos el archivo CSV y escribimos los encabezados
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss", "lattice_loss", "position_loss", "z_loss", "vram_mb", "epoch_time_sec"])

    # 5. Entrenamiento
    print(f"\n🔥 ¡Arrancando GPU! Entrenando hasta la época {EPOCHS}...\n")
    
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        
        # 📦 Almacenamiento temporal en RAM para TODOS los steps de esta época
        epoch_step_metrics = []
        batch_losses = [] # Solo para el postfix de tqdm
        
        pbar = tqdm(get_batches(dset, BATCH_SIZE, shuffle=True), total=total_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(pbar):
            try:
                graph = create_batched_dataset(
                    batch['atoms'], batch['positions'], batch['lattice'],
                    BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
                )
            except ValueError:
                continue
            
            target_lattice = jnp.array(batch['lattice'])
            target_pos = jnp.array(batch['positions'])
            target_z = jnp.array(batch['atoms'])
            
            state, loss, (l_lat, l_z, l_pos) = train_step(state, graph, target_lattice, target_pos, target_z)
            
            # Guardamos la info del step (excepto VRAM y Tiempo, eso va al final)
            epoch_step_metrics.append([
                epoch + 1,
                step + 1,
                float(loss.item()),
                float(l_lat.item()),
                float(l_pos.item()),
                float(l_z.item())
            ])
            
            batch_losses.append(loss.item())
            pbar.set_postfix({'Loss': f"{np.mean(batch_losses[-50:]):.4f}"})
                
        # --- FIN DE LA ÉPOCA ---
        epoch_time = time.time() - start_time
        avg_loss = float(np.mean(batch_losses))
        
        # Consultamos la VRAM SOLO UNA VEZ al terminar los cálculos
        current_vram = get_gpu_vram()
        
        # 💾 ESCRITURA EN BLOQUE AL CSV (Súper rápida)
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # A cada fila guardada, le agregamos la VRAM y el Tiempo total de la época
            for row in epoch_step_metrics:
                row.extend([current_vram, f"{epoch_time:.2f}"])
            # Escribimos cientos de filas en una sola operación de disco
            writer.writerows(epoch_step_metrics)

        # 6. Guardar Checkpoint
        checkpoint_manager.save(
            epoch + 1,
            args=ocp.args.PyTreeSave(state), 
            metrics=avg_loss
        )
        
        print(f"✅ Epoch {epoch+1} | Loss Avg: {avg_loss:.4f} | VRAM: {current_vram} | Logs CSV guardados.")

    checkpoint_manager.wait_until_finished()
    print("\n🎉 ENTRENAMIENTO FINALIZADO. Pesos guardados con éxito.")

if __name__ == "__main__":
    main()