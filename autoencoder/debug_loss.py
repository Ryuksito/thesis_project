# ==========================================
# SCRIPT DE DEPURACIÓN DE NANs (EAGER MODE)
# ==========================================
import os
import sys

if os.getcwd().endswith("autoencoder"):
    os.chdir("../")
sys.path.append(os.getcwd())    

import jax
import jax.numpy as jnp
import numpy as np
import optax

from autoencoder.config import *
from autoencoder.data.loader import load_dataset, get_batches, create_batched_dataset
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.loss import compute_lattice_loss, compute_z_loss, compute_positions_loss

# Desactivamos JIT globalmente para que los prints funcionen y los errores muestren la línea exacta
jax.config.update("jax_disable_jit", True)

def check_nan(tensor, name):
    """Función auxiliar para detectar NaNs instantáneamente."""
    if jnp.isnan(tensor).any():
        print(f"❌ ¡ALERTA! Se encontraron NaNs en: {name}")
        return True
    return False

def debug_step():
    print("\n" + "="*60)
    print("🐛 INICIANDO DEPURADOR DE LOSS (SIN JIT)")
    print("="*60)

    # 1. Cargar solo 1 batch de datos
    print("Cargando datos...")
    dset = load_dataset(DATA_PATH + "/autoencoder_16k.h5")
    print(f"dset.ids: {type(dset.ids), {dset.ids.dtype}}, {jnp.min(dset.ids)}, {jnp.max(dset.ids)}")
    print(f"dset.lattice: {type(dset.lattice), {dset.lattice.dtype}}, {jnp.min(dset.lattice)}, {jnp.max(dset.lattice)}")
    print(f"dset.positions: {type(dset.positions), {dset.positions.dtype}}, {jnp.min(dset.positions)}, {jnp.max(dset.positions)}")
    print(f"dset.atoms: {type(dset.atoms), {dset.atoms.dtype}}, {jnp.min(dset.atoms)}, {jnp.max(dset.atoms)}")

    batch = list(get_batches(dset, BATCH_SIZE, shuffle=True))[0]
    # 2. Inspección de Tensores de entrada
    print(f"\n🔍 INSPECCIÓN DE INPUTS (Raw Batch):")
    for key in ['atoms', 'positions', 'lattice']:
        data = jnp.array(batch[key])
        print(f"  -> {key}: Min={jnp.min(data):.4f}, Max={jnp.max(data):.4f}, NaN={jnp.isnan(data).any()}")
    
    graph = create_batched_dataset(
        batch['atoms'], batch['positions'], batch['lattice'],
        BATCH_SIZE, MAX_ATOMS, MAX_N_EDGES, MAX_ATOMIC_NUMBER, MAX_LATTICE_LENGTH, MAX_LATTICE_ANGLE, 5.0
    )

    # 4. Inspección del Grafo
    print(f"\n🔍 INSPECCIÓN DEL GRAFO (GraphsTuple):")
    print(f"  -> Nodes['Z']: Min={jnp.min(graph.nodes['Z'])}, Max={jnp.max(graph.nodes['Z'])}")
    print(f"  -> Nodes['pos']: Min={jnp.min(graph.nodes['pos']):.4f}, Max={jnp.max(graph.nodes['pos']):.4f}")

    if jnp.isinf(graph.nodes['pos']).any():
        print("❌ ¡ALERTA! Se encontraron Infinitos en las posiciones del grafo")
    
    target_lattice = jnp.array(batch['lattice'])
    target_pos = jnp.array(batch['positions'])
    target_z = jnp.array(batch['atoms'])
    
    print("\n🔍 INSPECCIÓN DE TARGETS:")
    print(f"Target Z Min/Max: {jnp.min(target_z)} / {jnp.max(target_z)} (Debe ser entero entre 0 y 118)")
    check_nan(target_lattice, "target_lattice")
    check_nan(target_pos, "target_pos")
    check_nan(target_z, "target_z")


    # 2. Inicializar Modelo
    key = jax.random.PRNGKey(42)
    model = Autoencoder(latent_dim=64, max_atoms=MAX_ATOMS)
    variables = model.init(key, graph)
    
    print("\n🧠 Ejecutando Forward Pass PASO A PASO...")
    
    # Vamos a capturar el estado interno del modelo
    params = variables['params']
    
    # 1. Probemos solo el Encoder
    latent = model.apply(variables, graph, method=model.encode)
    if check_nan(latent, "Latent Vector (Output Encoder)"):
        print("❌ ¡El Encoder está explotando!")
    
    # 2. Si el encoder está bien, probemos el Decoder
    pred_lattice_raw, pred_pos_raw, pred_z_logits_raw= model.apply(variables, latent, method=model.decode)

    # 3. Eliminar Grafo Basurero
    pred_lattice = pred_lattice_raw[:-1]
    pred_pos = pred_pos_raw[:-1]
    pred_z_logits = pred_z_logits_raw[:-1]
    
    if check_nan(pred_lattice, "Lattice (Decoder)"): print("❌ Lattice explota")
    if check_nan(pred_pos, "Pos (Decoder)"): print("❌ Pos explota")
    if check_nan(pred_z_logits, "Z Logits (Decoder)"): print("❌ Z Logits explota")
    
    
    
    print("\n🔍 INSPECCIÓN DE PREDICCIONES:")
    check_nan(pred_lattice, "pred_lattice")
    check_nan(pred_pos, "pred_pos")
    check_nan(pred_z_logits, "pred_z_logits")
    print(f"Logits Min/Max: {jnp.min(pred_z_logits):.4f} / {jnp.max(pred_z_logits):.4f}")

    # =================================================================
    # 4. CÁLCULO DE LOSS PASO A PASO
    # =================================================================
    print("\n📉 CALCULANDO LOSS COMPONENTE POR COMPONENTE...")
    
    # A) Lattice
    l_lat = compute_lattice_loss(pred_lattice, target_lattice)
    print(f"Lattice Loss: {l_lat}")
    check_nan(l_lat, "Lattice Loss")
    
    # B) Posiciones (ACSF)
    l_pos = compute_positions_loss(pred_pos, target_pos, target_z)
    print(f"Positions Loss (ACSF): {l_pos}")
    check_nan(l_pos, "Positions Loss")
    
    # C) Z (Cross-Entropy)
    try:
        l_z = compute_z_loss(pred_z_logits, target_z)
        print(f"Z Loss (Cross-Entropy): {l_z}")
        check_nan(l_z, "Z Loss")
    except Exception as e:
        print(f"❌ ERROR FATAL en Z Loss (Cross-Entropy): {e}")
        
    print("\n✅ Depuración finalizada.")

if __name__ == "__main__":
    debug_step()