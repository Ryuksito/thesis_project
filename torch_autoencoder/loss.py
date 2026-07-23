# autoencoder/loss.py
import torch
import torch.nn.functional as F

# =====================================================================
# 1. PÉRDIDA DE RED (LATTICE) -> Regresión Continua (MSE)
# =====================================================================
def compute_lattice_loss(pred_lattice, target_lattice):
    """
    Error cuadrático medio para los 6 parámetros de la celda unitaria.
    """
    return F.mse_loss(pred_lattice, target_lattice)

# =====================================================================
# 2. PÉRDIDA DE IDENTIDAD (Z) -> Clasificación Categórica
# =====================================================================
def compute_z_loss(pred_z_logits, target_z):
    """
    pred_z_logits: (Batch, Max_Atoms, 119)
    target_z: (Batch, Max_Atoms)
    """
    # F.cross_entropy espera (Batch, Classes, ...) -> Transponemos
    # reduction='none' para aplicar nuestras máscaras manuales
    ce_loss = F.cross_entropy(pred_z_logits.transpose(1, 2), target_z.long(), reduction='none')
    
    mask_real = (target_z > 0).float()
    mask_pad = 1.0 - mask_real
    
    num_real_atoms = mask_real.sum().clamp(min=1.0)
    num_pad_atoms = mask_pad.sum().clamp(min=1.0)
    
    loss_z_real = (ce_loss * mask_real).sum() / num_real_atoms
    loss_z_pad = (ce_loss * mask_pad).sum() / num_pad_atoms
    
    return loss_z_real + (0.5 * loss_z_pad)

# =====================================================================
# 3. PÉRDIDA DE POSICIONES -> Invarianza Permutacional (ACSF con PBC)
# =====================================================================
def compute_positions_loss(pred_pos, target_pos, target_z, eta=5.0):
    """
    Usa el espectro ordenado de distancias radiales locales.
    """
    mask_real = (target_z > 0).float()
    # (B, Max_Atoms, Max_Atoms)
    mask_2d = mask_real.unsqueeze(-1) * mask_real.unsqueeze(-2) 
    num_real_atoms = mask_real.sum().clamp(min=1.0)

    def get_sorted_fingerprints(pos):
        # Broadcasting para obtener matriz de diferencias de todos los pares
        # pos: (B, N, 3) -> diff: (B, N, N, 3)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        
        # Condiciones Periódicas de Contorno (PBC) en coordenadas fraccionales
        diff = diff - torch.round(diff)
        dist_sq = (diff ** 2).sum(dim=-1)
        
        gaussians = torch.exp(-eta * dist_sq) * mask_2d
        
        # Eliminar autoconexiones (diagonal)
        eye = torch.eye(pos.size(1), device=pos.device).unsqueeze(0)
        gaussians = gaussians * (1.0 - eye)
        
        fingerprints = gaussians.sum(dim=-1)
        
        # Sort es diferenciable en PyTorch. [0] devuelve los valores, [1] los índices.
        return torch.sort(fingerprints, dim=-1)[0]

    fp_pred = get_sorted_fingerprints(pred_pos)
    fp_target = get_sorted_fingerprints(target_pos)

    loss_pos = (((fp_pred - fp_target)**2) * mask_real).sum() / num_real_atoms
    return loss_pos

# =====================================================================
# ORQUESTADOR DE ENTRENAMIENTO (Total Loss)
# =====================================================================
def compute_total_loss(pred_lattice, pred_pos, pred_z_logits, target_lattice, target_pos, target_z):
    l_lat = compute_lattice_loss(pred_lattice, target_lattice)
    l_z = compute_z_loss(pred_z_logits, target_z)
    l_pos = compute_positions_loss(pred_pos, target_pos, target_z)
    
    # Pesos empíricos heredados
    w_lat = 100.0 * l_lat
    w_z = 1.0 * l_z
    w_pos = 40.0 * l_pos 
    
    total_loss = w_lat + w_z + w_pos
    
    # 🔥 CORRECCIÓN: Devolvemos tensores crudos. Cero llamadas a .item() aquí.
    return total_loss, (l_lat, l_z, l_pos)

def mse_loss(pred_lattice, pred_pos, pred_z, target_lattice, target_pos, target_z):    
    mask_real = (target_z > 0).float()
    mask_pad = 1.0 - mask_real
    mask_expanded = mask_real.unsqueeze(-1)
    
    num_real_atoms = mask_real.sum().clamp(min=1.0)
    num_pad_atoms = mask_pad.sum().clamp(min=1.0)

    loss_lattice = F.mse_loss(pred_lattice, target_lattice)
    
    mse_z_real = ((pred_z - target_z)**2) * mask_real
    loss_z_real = mse_z_real.sum() / num_real_atoms
    
    mse_z_pad = (pred_z**2) * mask_pad
    loss_z_pad = mse_z_pad.sum() / num_pad_atoms
    
    loss_z = loss_z_real + (0.5 * loss_z_pad) 
    
    mse_pos = ((pred_pos - target_pos)**2) * mask_expanded
    loss_pos = mse_pos.sum() / num_real_atoms
    
    total_loss = (1.0 * loss_lattice) + (1.0 * loss_z) + (10.0 * loss_pos)
    
    # 🔥 CORRECCIÓN: Devolvemos tensores crudos. Cero llamadas a .item() aquí.
    return total_loss, (loss_lattice, loss_z, loss_pos)

# =====================================================================
# PÉRDIDA SUPERVIVENCIA NEAT
# =====================================================================
def crystal_loss_fn(preds, y_target, lattice_params=6, max_atoms=24, separate_results=False):
    """
    Filtro Geométrico Invariante para el pipeline evolutivo.
    """
    pred_lat = preds[:, :lattice_params]
    target_lat = y_target[:, :lattice_params]
    
    pred_atoms = preds[:, lattice_params:].view(-1, max_atoms, 4)
    target_atoms = y_target[:, lattice_params:].view(-1, max_atoms, 4)
    
    pred_z = pred_atoms[:, :, 0]
    target_z = target_atoms[:, :, 0]
    
    mask = (target_z > 0).float()
    mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    
    # 1. LATTICE
    lat_loss = F.mse_loss(pred_lat, target_lat)

    # 2. ACSF
    def get_sorted_fingerprints(pos, m2d, eta=5.0):
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        diff = diff - torch.round(diff)
        dist_sq = (diff**2).sum(dim=-1)
        gaussians = torch.exp(-eta * dist_sq) * m2d
        
        eye = torch.eye(pos.size(1), device=pos.device).unsqueeze(0)
        gaussians = gaussians * (1.0 - eye)
        
        fingerprints = gaussians.sum(dim=-1)
        return torch.sort(fingerprints, dim=-1)[0]

    fp_pred = get_sorted_fingerprints(pred_atoms[:, :, 1:], mask_2d)
    fp_target = get_sorted_fingerprints(target_atoms[:, :, 1:], mask_2d)
    acsf_loss = (((fp_pred - fp_target)**2) * mask).sum() / mask.sum().clamp(min=1.0)

    # 3. Z-LOSS INVARIANTE
    sorted_pred_z = torch.sort(pred_z * mask, dim=1)[0]
    sorted_target_z = torch.sort(target_z, dim=1)[0]
    z_loss = F.mse_loss(sorted_pred_z, sorted_target_z)

    # 4. VARIANCE & REPULSION
    fp_var_pred = torch.var(fp_pred * mask, dim=0, unbiased=False)
    fp_var_target = torch.var(fp_target * mask, dim=0, unbiased=False)
    var_loss = F.mse_loss(fp_var_pred, fp_var_target)
    
    repulsion = torch.exp(-1000.0 * (fp_pred + 1e-6)) * mask
    repulsion_loss = repulsion.sum() / mask.sum().clamp(min=1.0)

    w_lat = 10.0 * lat_loss
    w_acsf = 50.0 * acsf_loss
    w_z = 20.0 * z_loss
    w_var = 10.0 * var_loss
    w_rep = 10.0 * repulsion_loss

    total_loss = w_lat + w_acsf + w_z + w_var + w_rep
    
    if separate_results:
        return total_loss, w_lat, w_acsf, w_z, w_var, w_rep
    
    return total_loss