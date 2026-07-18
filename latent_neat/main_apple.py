# ==========================================
# CONFIGURACIÓN DE RUTAS (Root Execution)
# ==========================================
import gc
import os
import sys
import time
import json
import csv
import subprocess

# Optimizaciones extremas de memoria para JAX/XLA
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['JAX_PLATFORMS'] = 'cpu'

if os.getcwd().endswith("latent_neat"):
    os.chdir("../")
sys.path.append(os.getcwd())    

print(f"Current working directory: {os.getcwd()}")
print(f"Python import paths: {sys.path[-1]}")

import optax
import orbax.checkpoint as ocp
import numpy as np
import jax
import jax.numpy as jnp
from tensorneat import algorithm, genome
from tensorneat.common.functions import act_jnp
from tensorneat.genome import DefaultMutation
from tensorneat.common import State
from tqdm.auto import tqdm

from autoencoder import config as autoencoder_config
from latent_neat import config
from latent_neat.data.loader import load_dataset, BatchLoader
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.autoencoder import load_decoder

# --- RUTAS ---
# LOGS_DIR = "/home/alanh/Dev/owns/thesis_project/latent_neat/runs/v1/"
BASE_DIR = os.getcwd()
LOGS_DIR = os.path.join(BASE_DIR, "latent_neat", "runs", "v2")
print(f"Los logs se guardarán en: {LOGS_DIR}")
CHKPT_DIR = os.path.join(LOGS_DIR, "checkpoints")
# AUTOENCODER_LOGS_DIR = "/home/alanh/Dev/owns/thesis_project/autoencoder/runs/v1/"
AUTOENCODER_LOGS_DIR = os.path.join(BASE_DIR, "autoencoder", "runs", "v2")
AUTOENCODER_CHECKPOINT_DIR = os.path.join(AUTOENCODER_LOGS_DIR, "checkpoints")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHKPT_DIR, exist_ok=True)

# ── FUNCIONES AUXILIARES ───────────────────────────────────────────────────

def get_gpu_vram():
    try:
        res = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return float(res.strip().split('\n')[0]) / 1024.0 # Retorna en GB
    except Exception:
        return 0.0

@jax.jit
def get_lr(step):
    decay_ratio = step / config.TOTAL_GRAD_STEPS
    coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
    return config.LR_MIN + coeff * (config.LR_MAX - config.LR_MIN)

# ── 1. SETUP DE DATOS Y JUECES ─────────────────────────────────────────────

dset_train, dset_test = load_dataset(os.path.join(config.DATA_PATH, "latent-neat-dataset.h5"))
train_loader = BatchLoader(dset_train, config.BATCH_SIZE, shuffle=True)

print(f"📥 Cargando Juez Autoencoder categórico (V2)...")
decoder_model, decoder_params = load_decoder(AUTOENCODER_LOGS_DIR)

# ── 2. SETUP DE TENSORNEAT ─────────────────────────────────────────────────

neat = algorithm.NEAT(
    pop_size=config.POPSIZE,
    species_size=config.SPECIES_SIZE,
    survival_threshold=config.SURVIVAL_THRESHOLD,
    pop_batch_size=config.POP_BATCH_SIZE,
    genome=genome.DefaultGenome(
        num_inputs=config.INPUT_DIM,     
        num_outputs=config.OUTPUT_DIM,   
        max_nodes=config.MAX_NODES,
        max_conns=config.MAX_CONNS,      
        init_hidden_layers=(16,),
        output_transform=act_jnp.sigmoid_, 
        mutation=DefaultMutation(
            conn_add=config.CONN_ADD_PROB,
            conn_delete=config.CONN_DELETE_PROB,
            node_add=config.NODE_ADD_PROB,
            node_delete=config.NODE_DELETE_PROB,
        ),
    ),
)

state = State(randkey=jax.random.key(config.SEED))
state = neat.setup(state)
g = neat.genome

# ── 3. LÓGICA DE JAX COMPILADA ─────────────────────────────────────────────
@jax.jit
def compute_z_loss(pred_z_logits, target_z):
    """
    Evalúa la identidad del átomo usando Cross-Entropy (119 clases).
    pred_z_logits: (Batch, Max_Atoms, 119)
    target_z: (Batch, Max_Atoms) con números enteros (0 a 118)
    """
    # Convertimos a entero para Optax
    target_z_int = target_z.astype(jnp.int32)
    
    # Máscaras
    mask_real = (target_z > 0).astype(jnp.float32)
    mask_pad = 1.0 - mask_real
    
    num_real_atoms = jnp.maximum(jnp.sum(mask_real), 1.0)
    num_pad_atoms = jnp.maximum(jnp.sum(mask_pad), 1.0)
    
    # Cross-Entropy Matemática Pura
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=pred_z_logits, 
        labels=target_z_int
    )
    
    # Aplicamos máscaras
    loss_z_real = jnp.sum(ce_loss * mask_real) / num_real_atoms
    loss_z_pad = jnp.sum(ce_loss * mask_pad) / num_pad_atoms
    
    # Priorizamos aprender átomos reales, pero penalizamos levemente predecir padding mal
    return loss_z_real + (0.5 * loss_z_pad)

@jax.jit
def compute_loss(neat_preds, target_lattice, target_pos, target_z, decoder_params):
    pred_lattice, pred_pos, pred_z = decoder_model.apply(decoder_params, neat_preds)

    l_lat = jnp.mean(jnp.square(pred_lattice - target_lattice))
    l_z = compute_z_loss(pred_z, target_z)

    mask = (target_z > 0.0).astype(jnp.float32)
    mask_expanded = jnp.expand_dims(mask, -1)
    mse_pos = jnp.square(pred_pos - target_pos) * mask_expanded
    l_pos = jnp.sum(mse_pos) / jnp.maximum(jnp.sum(mask_expanded), 1.0)

    total = (1.0 * l_lat) + (1.0 * l_z) + (10.0 * l_pos)
    return total, l_lat, l_z, l_pos

def grad_step(nodes, conns, state, batch_inputs, target_lattice, target_pos, target_z, lr, decoder_params):
    def loss_fn(preds):
        tot, _, _, _ = compute_loss(preds, target_lattice, target_pos, target_z, decoder_params)
        return tot

    loss, (grads_n, grads_c) = g.grad(state, nodes, conns, batch_inputs, loss_fn)
    
    # Clipping de gradientes (Evita explosiones matemáticas)
    grads_n = jnp.clip(grads_n, -1.0, 1.0)
    grads_c = jnp.clip(grads_c, -1.0, 1.0)

    return nodes - lr * grads_n, conns - lr * grads_c, loss

def evaluate_step(nodes, conns, state, batch_inputs, target_lattice, target_pos, target_z, decoder_params):
    # Transforma genoma a red usable
    transformed_network = g.transform(state, nodes, conns)
    # Forward pass sobre todo el batch (vmap)
    preds = jax.vmap(g.forward, in_axes=(None, None, 0))(state, transformed_network, batch_inputs)
    # Evaluamos físicamente en el decoder
    return compute_loss(preds, target_lattice, target_pos, target_z, decoder_params)

# Compilación Masiva Vectorizada
batch_grad_step = jax.jit(jax.vmap(grad_step, in_axes=(0, 0, None, None, None, None, None, None, None)))
batch_evaluate = jax.jit(jax.vmap(evaluate_step, in_axes=(0, 0, None, None, None, None, None, None)))

print("✅ Motores JAX Compilados.")

# ── 4. BUCLE EVOLUTIVO DE BALDWIN (V1 Style) ───────────────────────────────

json_log_path = os.path.join(LOGS_DIR, "neat_history.json")
history_logs = {}

pbar = tqdm(range(config.N_GENERATIONS), desc="🧬 Evolución NEAT + JAX")

for generation in pbar:
    start_time = time.time()
    
    # 1. NEAT genera la población base para esta generación
    pop_nodes, pop_conns = neat.ask(state)

    # Obtenemos el batch de esta generación para entrenar
    batch = next(train_loader)
    batch_inputs = jnp.concatenate([
        jnp.array(batch["context_elements"]),
        jnp.array(batch["context_embeddings"]),
        jnp.array(batch["context_props"])
    ], axis=-1)

    target_lattice = jnp.array(batch["target_lattice"])
    target_pos = jnp.array(batch["target_positions"])
    target_z = jnp.array(batch["target_atoms"])

    gen_lr = float(get_lr(generation * config.GRAD_STEPS_PER_GEN))
    loss_estudio_inicio = float('inf')
    loss_estudio_final = float('inf')

    # 2. Descenso de Gradiente (El Individuo "aprende" durante su vida)
    for step in range(config.GRAD_STEPS_PER_GEN):
        global_step = generation * config.GRAD_STEPS_PER_GEN + step
        current_lr = get_lr(global_step)
        
        pop_nodes, pop_conns, batch_losses = batch_grad_step(
            pop_nodes, pop_conns, state, 
            batch_inputs, target_lattice, 
            target_pos, target_z, current_lr,
            decoder_params
        )
        
        if step == 0:
            loss_estudio_inicio = float(np.nanmin(jax.device_get(batch_losses)))
        if step == config.GRAD_STEPS_PER_GEN - 1:
            loss_estudio_final = float(np.nanmin(jax.device_get(batch_losses)))

    # 3. Evaluación Final para Selección Darwiniana
    # ¿Qué topología fue la mejor "aprendiendo"?
    cpu_totals, cpu_lats, cpu_zs, cpu_pos = jax.device_get(
        batch_evaluate(pop_nodes, pop_conns, state, batch_inputs, target_lattice, target_pos, target_z, decoder_params)
    )

    valid_mask = np.isfinite(cpu_totals)
    cpu_losses_safe = np.where(valid_mask, cpu_totals, 1e6)
    
    # Fitness = negativo del loss (NEAT maximiza)
    fitnesses = -cpu_losses_safe

    # --- 🧹 LIMPIEZA MANUAL ANTES DEL CROSSOVER ---
    del batch, batch_inputs, target_lattice, target_pos, 
    gc.collect()
    
    # 4. NEAT selecciona y genera los padres de la sig. generación
    state = neat.tell(state, fitnesses)

    # 5. Telemetría y Logging Dinámico
    valid_count = int(valid_mask.sum())
    best_idx = int(np.nanargmin(cpu_losses_safe))
    
    bl_total = float(cpu_totals[best_idx]) if valid_count > 0 else float('nan')
    bl_lat = float(cpu_lats[best_idx])
    bl_z = float(cpu_zs[best_idx])
    bl_pos = float(cpu_pos[best_idx])
    
    peak_gb = get_gpu_vram()
    gen_time = time.time() - start_time
    
    pbar.set_postfix({
        'InitLoss': f"{loss_estudio_inicio:.2f}",
        'FinalLoss': f"{loss_estudio_final:.2f}",
        'BestEval': f"{bl_total:.2f}",
        'VRAM': f"{peak_gb:.1f}G"
    })

    history_logs[str(generation)] = {
        "learning_rate": gen_lr,
        "init_grad_loss": loss_estudio_inicio,
        "final_grad_loss": loss_estudio_final,
        "best_eval_loss": bl_total,
        "best_lattice_loss": bl_lat,
        "best_z_loss": bl_z,
        "best_pos_loss": bl_pos,
        "valid_pop": valid_count,
        "vram_peak_gb": peak_gb,
        "time_sec": gen_time
    }

    # Guardar en disco
    if generation % 10 == 0 or generation == config.N_GENERATIONS - 1:
        with open(json_log_path, 'w') as f:
            json.dump(history_logs, f, indent=4)
            
        best_nodes = jax.device_get(pop_nodes[best_idx])
        best_conns = jax.device_get(pop_conns[best_idx])
        
        chkpt_file = os.path.join(CHKPT_DIR, f"best_gen_{generation:04d}.npz")
        np.savez(chkpt_file, nodes=best_nodes, conns=best_conns, loss=bl_total)

print("\n✅ Entrenamiento Híbrido Finalizado. Checkpoints guardados en:", CHKPT_DIR)