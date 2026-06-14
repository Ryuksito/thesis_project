import jax
import jax.numpy as jnp
import optax

# =====================================================================
# 1. PÉRDIDA DE RED (LATTICE) -> Regresión Continua (MSE)
# =====================================================================
@jax.jit
def compute_lattice_loss(pred_lattice, target_lattice):
    """
    Error cuadrático medio para los 6 parámetros de la celda unitaria.
    """
    return jnp.mean(jnp.square(pred_lattice - target_lattice))

# =====================================================================
# 2. PÉRDIDA DE IDENTIDAD (Z) -> Clasificación Categórica (Cross-Entropy)
# =====================================================================
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

# =====================================================================
# 3. PÉRDIDA DE POSICIONES -> Invarianza Permutacional (ACSF con PBC)
# =====================================================================
@jax.jit
def compute_positions_loss(pred_pos, target_pos, target_z, eta=5.0):
    """
    Evalúa las coordenadas XYZ usando ACSF para evitar el problema de índices.
    Las coordenadas que estén fuera de orden pero formen la misma figura darán loss = 0.
    """
    mask_real = (target_z > 0).astype(jnp.float32)
    mask_2d = mask_real[:, :, None] * mask_real[:, None, :]
    num_real_atoms = jnp.maximum(jnp.sum(mask_real), 1.0)

    # Función interna para calcular huellas dactilares (fingerprints)
    def get_sorted_fingerprints(pos):
        # Distancia entre todos los pares de átomos
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        # Condiciones Periódicas de Contorno (PBC)
        diff = diff - jnp.round(diff) 
        dist_sq = jnp.sum(diff**2, axis=-1)
        
        # Filtro Gaussiano
        gaussians = jnp.exp(-eta * dist_sq) * mask_2d
        
        # Ignorar la distancia de un átomo consigo mismo
        eye = jnp.eye(pos.shape[1])[None, :, :]
        gaussians = gaussians * (1.0 - eye)
        
        # Sumar influencias y ordenar para lograr invarianza permutacional
        fingerprints = jnp.sum(gaussians, axis=-1)
        return jnp.sort(fingerprints, axis=-1)

    fp_pred = get_sorted_fingerprints(pred_pos)
    fp_target = get_sorted_fingerprints(target_pos)

    # El MSE se aplica sobre la huella geométrica, no sobre las coordenadas crudas
    loss_pos = jnp.sum(((fp_pred - fp_target)**2) * mask_real) / num_real_atoms
    return loss_pos

# =====================================================================
# ORQUESTADOR DE ENTRENAMIENTO (Total Loss)
# =====================================================================
@jax.jit
def compute_total_loss(pred_lattice, pred_pos, pred_z_logits, target_lattice, target_pos, target_z):
    """
    Función unificada para ser llamada por grad_fn o train_step.
    (Nota: Ya no hay recortes [:-1] aquí. Se asume que la entrada está limpia).
    """
    # 1. Calculamos las pérdidas individuales
    l_lat = compute_lattice_loss(pred_lattice, target_lattice)
    l_z = compute_z_loss(pred_z_logits, target_z)
    l_pos = compute_positions_loss(pred_pos, target_pos, target_z)
    
    # 2. Pesos del balance multimodal
    # Ajusta estos multiplicadores si notas que una métrica domina a las demás
    w_lat = 1.0 * l_lat
    w_z = 1.0 * l_z
    w_pos = 10.0 * l_pos 
    
    total_loss = w_lat + w_z + w_pos
    
    # Devolvemos el total para el optimizador y la tupla para los logs
    return total_loss, (l_lat, l_z, l_pos)

def mse_loss(pred_lattice, pred_pos, pred_z, target_lattice, target_pos, target_z):    
    # MÁSCARAS CRÍTICAS PARA CRISTALES
    mask_real = (target_z > 0.0).astype(jnp.float32)  # 1 donde hay átomo, 0 en padding
    mask_pad = 1.0 - mask_real                        # 1 donde hay padding, 0 en átomo
    mask_expanded = jnp.expand_dims(mask_real, -1)
    
    num_real_atoms = jnp.maximum(jnp.sum(mask_real), 1.0)
    num_pad_atoms = jnp.maximum(jnp.sum(mask_pad), 1.0)

    # 2. Error del Lattice (El lattice afecta a todo el cristal, no requiere máscara)
    loss_lattice = jnp.mean(jnp.square(pred_lattice - target_lattice))
    
    # 3. Error de Identidad Z (CORREGIDO)
    # A) ¿Qué tan bien adivina los átomos reales?
    mse_z_real = jnp.square(pred_z - target_z) * mask_real
    loss_z_real = jnp.sum(mse_z_real) / num_real_atoms
    
    # B) ¿Qué tan bien mantiene en cero el padding? (Lo calculamos por separado)
    mse_z_pad = jnp.square(pred_z) * mask_pad
    loss_z_pad = jnp.sum(mse_z_pad) / num_pad_atoms
    
    # Sumamos ambos, pero le damos prioridad a adivinar los átomos reales
    loss_z = loss_z_real + (0.5 * loss_z_pad) 
    
    # 4. Error de Posiciones (Ya lo tenías bien, pero aseguramos la división correcta)
    mse_pos = jnp.square(pred_pos - target_pos) * mask_expanded
    loss_pos = jnp.sum(mse_pos) / num_real_atoms
    
    # 5. Suma Ponderada (Balance de Gradientes)
    total_loss = (1.0 * loss_lattice) + (1.0 * loss_z) + (10.0 * loss_pos)
    
    return total_loss, (loss_lattice, loss_z, loss_pos)

@jax.jit
def crystal_loss_fn(preds, y_target, lattice_params=6, max_atoms=24, separate_results=False):
    """
    Pérdida para la Supervivencia NEAT (Filtro Geométrico Invariante).
    """
    pred_lat = preds[:, :lattice_params]
    target_lat = y_target[:, :lattice_params]
    
    pred_atoms = preds[:, lattice_params:].reshape(-1, max_atoms, 4)
    target_atoms = y_target[:, lattice_params:].reshape(-1, max_atoms, 4)
    
    pred_z = pred_atoms[:, :, 0]
    target_z = target_atoms[:, :, 0]
    
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

    fp_pred = get_sorted_fingerprints(pred_atoms[:, :, 1:], mask_2d)
    fp_target = get_sorted_fingerprints(target_atoms[:, :, 1:], mask_2d)
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
    
    if separate_results:
        return total_loss, w_lat, w_acsf, w_z, w_var, w_rep
    
    return total_loss