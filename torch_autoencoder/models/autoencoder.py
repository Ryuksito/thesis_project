# autoencoder/models/autoencoder.py
import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.utils import scatter
from torch_geometric.nn import global_add_pool

# ==============================================================================
# 1. FUNCIONES FÍSICAS (Vectorizadas para PyTorch)
# ==============================================================================
def lattice_to_matrix_torch(params):
    """Convierte parámetros de celda a matriz cartesiana (Soporta Batches)"""
    a, b, c = params[..., 0], params[..., 1], params[..., 2]
    alpha, beta, gamma = params[..., 3], params[..., 4], params[..., 5]
    alpha_r, beta_r, gamma_r = torch.deg2rad(alpha), torch.deg2rad(beta), torch.deg2rad(gamma)
    
    va = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vb = torch.stack([b * torch.cos(gamma_r), b * torch.sin(gamma_r), torch.zeros_like(b)], dim=-1)
    
    cx = c * torch.cos(beta_r)
    cy = c * (torch.cos(alpha_r) - torch.cos(beta_r) * torch.cos(gamma_r)) / (torch.sin(gamma_r) + 1e-7)
    cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-7))
    vc = torch.stack([cx, cy, cz], dim=-1)
    
    return torch.stack([va, vb, vc], dim=-2)

def radial_basis(distances, num_basis=32, cutoff=5.0):
    centers = torch.linspace(0.0, cutoff, num_basis, device=distances.device)
    gamma = 1.0 / (cutoff / num_basis)**2
    # Distances: (E) -> (E, 1) - centers (32) -> (E, 32)
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers)**2)

# ==============================================================================
# 2. EL ENCODER EQUIVARIANTE (e3nn PyTorch)
# ==============================================================================
class CrystalEncoder(nn.Module):
    def __init__(self, latent_dim=64, max_atomic_number=118):
        super().__init__()
        self.latent_dim = latent_dim
        vocab_size = max_atomic_number + 1
        
        self.node_embed = nn.Embedding(vocab_size, 64)
        
        # e3nn setup
        self.irreps_in = o3.Irreps("64x0e")
        self.irreps_sh = o3.Irreps("1x0e + 1x1o + 1x2e")
        self.irreps_out = o3.Irreps("64x0e")
        
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, 
            self.irreps_sh, 
            self.irreps_out,
            internal_weights=False,
            shared_weights=False
        )
        self.radial_weights = nn.Linear(32, self.tp.weight_numel)
        
        self.latent_proj = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, data):
        # En PyG, 'data' contiene todo el batch concatenado
        edge_src, edge_dst = data.edge_index
        
        # Mapear las celdas unitarias (Batch) a cada arista usando data.batch
        cells = lattice_to_matrix_torch(data.lattice) 
        edge_cells = cells[data.batch[edge_src]] 
        
        # Vectores Cartesianos con PBC
        diff_frac = data.pos[edge_dst] - data.pos[edge_src] + data.edge_shift
        diff_cart = torch.einsum('ei,eij->ej', diff_frac, edge_cells)
        
        distances = torch.norm(diff_cart, dim=-1)

        safe_diff_cart = torch.where(
            distances.unsqueeze(-1) > 1e-6, 
            diff_cart, 
            torch.tensor([1.0, 0.0, 0.0], device=diff_cart.device)
        )
        
        # NOTA: Todo el "blindaje" desaparece. No hay aristas de padding.
        edge_rbf = radial_basis(distances)
        sh = o3.spherical_harmonics(self.irreps_sh, safe_diff_cart, normalize=True, normalization='component')
        
        # Embedding y Convolución
        node_features = self.node_embed(data.x)
        weights = self.radial_weights(edge_rbf)
        messages = self.tp(node_features[edge_src], sh, weights)
        
        # Agregación en nodos destino
        node_updates = scatter(messages, edge_dst, dim=0, dim_size=data.num_nodes, reduce='sum')
        
        # Pooling global al vector latente
        global_features = global_add_pool(node_updates, data.batch)
        latent = self.latent_proj(global_features)
        
        return latent

# ==============================================================================
# 3. EL DECODER GENERATIVO
# ==============================================================================
class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim=64, max_atoms=24, max_atomic_number=118):
        super().__init__()
        self.max_atoms = max_atoms
        vocab_size = max_atomic_number + 1
        
        self.lattice_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Softplus()
        )
        
        self.pos_emb = nn.Embedding(max_atoms, 32)
        
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.pos_head = nn.Sequential(
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
        self.z_head = nn.Linear(256, vocab_size)

    def forward(self, latent_vector):
        batch_size = latent_vector.size(0)
        
        # Salida A
        pred_lattice = self.lattice_net(latent_vector)
        
        # Positional Encoding expandido para todo el batch
        idx = torch.arange(self.max_atoms, device=latent_vector.device)
        pos_emb = self.pos_emb(idx).unsqueeze(0).expand(batch_size, -1, -1) 
        
        # Expandir latente (Batch, 1, Dim) -> (Batch, 24, Dim)
        latent_expanded = latent_vector.unsqueeze(1).expand(-1, self.max_atoms, -1)
        
        x_local = torch.cat([latent_expanded, pos_emb], dim=-1)
        x_local = self.shared_net(x_local)
        
        # Salidas B y C
        pred_pos = self.pos_head(x_local)
        pred_z_logits = self.z_head(x_local)
        
        return pred_lattice, pred_pos, pred_z_logits

# ==============================================================================
# 4. AUTOENCODER ORQUESTADOR Y UTILIDADES DE VRAM
# ==============================================================================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64, max_atoms=24, max_atomic_number=118):
        super().__init__()
        self.encoder = CrystalEncoder(latent_dim, max_atomic_number)
        self.decoder = CrystalDecoder(latent_dim, max_atoms, max_atomic_number)
        
    def forward(self, data):
        latent_vector = self.encoder(data)
        pred_lattice, pred_pos, pred_z = self.decoder(latent_vector)
        return pred_lattice, pred_pos, pred_z, latent_vector

# --- REEMPLAZO DE LA CIRUGÍA VRAM DE JAX ---
def load_decoder_only(ckpt_path: str, latent_dim=64, max_atoms=24, max_atomic_number=118, device='cpu'):
    """
    Carga de pesos quirúrgica nativa de PyTorch.
    El Encoder ni siquiera se instancia en RAM.
    """
    print(f"📥 Aislado Decoder desde {ckpt_path}...")
    decoder = CrystalDecoder(latent_dim, max_atoms, max_atomic_number).to(device)
    
    # Cargamos estado completo
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Filtramos SOLO las llaves que empiezan con 'decoder.' y quitamos ese prefijo
    decoder_state = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
    
    decoder.load_state_dict(decoder_state)
    print("✅ Decoder aislado y montado exitosamente.")
    return decoder