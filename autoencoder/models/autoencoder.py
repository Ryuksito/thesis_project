import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
import e3nn_jax as e3nn
import config

# ==============================================================================
# FUNCIONES FÍSICAS
# ==============================================================================
def lattice_to_matrix_jnp(params):
    a, b, c = params[..., 0], params[..., 1], params[..., 2]
    alpha, beta, gamma = params[..., 3], params[..., 4], params[..., 5]
    alpha_r, beta_r, gamma_r = jnp.radians(alpha), jnp.radians(beta), jnp.radians(gamma)
    
    va = jnp.stack([a, jnp.zeros_like(a), jnp.zeros_like(a)], axis=-1)
    vb = jnp.stack([b * jnp.cos(gamma_r), b * jnp.sin(gamma_r), jnp.zeros_like(b)], axis=-1)
    
    cx = c * jnp.cos(beta_r)
    cy = c * (jnp.cos(alpha_r) - jnp.cos(beta_r) * jnp.cos(gamma_r)) / (jnp.sin(gamma_r) + 1e-7)
    cz = jnp.sqrt(jnp.maximum(1e-7, c**2 - cx**2 - cy**2))
    vc = jnp.stack([cx, cy, cz], axis=-1)
    
    return jnp.stack([va, vb, vc], axis=-2)

def radial_basis(x, num_basis=32, cutoff=5.0):
    centers = jnp.linspace(0.0, cutoff, num_basis)
    gamma = 1.0 / (cutoff / num_basis)**2
    return jnp.exp(-gamma * (x - centers)**2)

# ==============================================================================
# 1. EL ENCODER EQUIVARIANTE (e3nn-jax) CON GRAPH MASKING
# ==============================================================================
class CrystalEncoder(nn.Module):
    latent_dim: int = 64
    max_atomic_number: int = 118
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        # A) Construcción de Vectores Cartesianos con PBC
        node_graph_indices = jnp.repeat(
            jnp.arange(graph.n_node.shape[0]), 
            graph.n_node, 
            total_repeat_length=graph.nodes["pos"].shape[0]
        )
        edge_graph_indices = node_graph_indices[graph.senders]
        
        cells = lattice_to_matrix_jnp(graph.globals["lattice"])
        edge_cells = cells[edge_graph_indices]
        
        diff_frac = graph.nodes["pos"][graph.receivers] - graph.nodes["pos"][graph.senders] + graph.edges["shifts"]
        diff_cart = jnp.einsum('ei,eij->ej', diff_frac, edge_cells) 
        
        # B) Armónicos Esféricos y Distancia Radial
        distances = jnp.linalg.norm(diff_cart, axis=-1, keepdims=True)
        
        # --- 🛡️ INICIO DEL BLINDAJE (MASKING) 🛡️ ---
        # 1. Crear máscara: 1.0 para aristas reales, 0.0 para aristas de padding
        edge_mask = (distances > 1e-6).astype(jnp.float32)
        
        # 2. Vector seguro: Reemplaza los vectores [0,0,0] por [1,0,0] SOLO en el padding
        # Esto evita que e3nn divida por cero al normalizar
        safe_diff_cart = jnp.where(distances > 1e-6, diff_cart, jnp.array([1.0, 0.0, 0.0]))
        
        # 3. Aplicar máscara a la distancia y usar vector seguro para SH
        edge_rbf = radial_basis(distances) * edge_mask 
        sh = e3nn.spherical_harmonics("0e + 1o + 2e", safe_diff_cart, normalize=True)
        # --- FIN DEL BLINDAJE ---
        
        # C) Embedding Químico (Nodo)
        vocab_size = int(self.max_atomic_number) + 1 
        z_indices = graph.nodes["Z"].astype(jnp.int32) 
        node_embed = nn.Embed(num_embeddings=vocab_size, features=64)(z_indices)
        node_features = e3nn.IrrepsArray("64x0e", node_embed)

        # D) Convolución Equivariante
        radial_weights = nn.Dense(64)(edge_rbf)
        weighted_array = node_features[graph.senders].array * radial_weights
        weighted_senders = e3nn.IrrepsArray("64x0e", weighted_array)
        
        messages = e3nn.tensor_product(weighted_senders, sh)
        
        # --- 🛡️ BLINDAJE FINAL 🛡️ ---
        # 4. Multiplicamos los mensajes por la máscara antes de sumar.
        # Así aseguramos que el grafo fantasma envíe "0" a los nodos reales.
        messages_array = messages.array * edge_mask
        
        # Sumamos la información de forma segura
        node_updates = jax.ops.segment_sum(messages_array, graph.receivers, num_segments=graph.nodes["Z"].shape[0])
        node_features = e3nn.IrrepsArray(messages.irreps, node_updates)
        
        # E) Pooling
        scalar_features = node_features.filter("0e").array
        global_features = jax.ops.segment_sum(scalar_features, node_graph_indices, num_segments=graph.n_node.shape[0])
        latent = nn.tanh(nn.Dense(self.latent_dim)(global_features))
        
        return latent

# ==============================================================================
# 2. EL DECODER GENERATIVO (Flax MLP)
# ==============================================================================
class CrystalDecoder(nn.Module):
    max_atoms: int = 24
    max_atomic_number: int = 118 # Se recibe del Autoencoder
    
    @nn.compact
    def __call__(self, latent_vector):
        x = nn.relu(nn.Dense(256)(latent_vector))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(512)(x))
        
        # SALIDA A: 6 Parámetros de Red positivos (Regresión MSE continua)
        pred_lattice = nn.softplus(nn.Dense(6)(x))
        
        # SALIDA B: 72 Coordenadas fraccionales [0, 1] (Regresión MSE continua)
        pred_pos_flat = nn.sigmoid(nn.Dense(self.max_atoms * 3)(x))
        pred_pos = pred_pos_flat.reshape(-1, self.max_atoms, 3)
        
        # SALIDA C: Identidades Atómicas - CORREGIDO A CLASIFICACIÓN
        vocab_size = int(self.max_atomic_number) + 1 # 119 casilleros posibles
        
        # Ya NO usamos ReLU. Generamos Logits puros (probabilidades matemáticas)
        pred_z_logits = nn.Dense(self.max_atoms * vocab_size)(x)
        # La forma será: (Batch, 24 átomos, 119 probabilidades)
        pred_z = pred_z_logits.reshape(-1, self.max_atoms, vocab_size)
        
        return pred_lattice, pred_pos, pred_z

# ==============================================================================
# 3. AUTOENCODER ORQUESTADOR
# ==============================================================================
class Autoencoder(nn.Module):
    latent_dim: int = config.LATENT_DIM
    max_atoms: int = config.MAX_ATOMS
    max_atomic_number: int = config.MAX_ATOMIC_NUMBER
    
    def setup(self):
        # Repartimos el parámetro a los hijos
        self.encoder = CrystalEncoder(
            latent_dim=self.latent_dim, 
            max_atomic_number=self.max_atomic_number
        )
        self.decoder = CrystalDecoder(
            max_atoms=self.max_atoms, 
            max_atomic_number=self.max_atomic_number
        )
        
    def __call__(self, graph: jraph.GraphsTuple):
        latent_vector = self.encoder(graph)
        pred_lattice, pred_pos, pred_z = self.decoder(latent_vector)
        return pred_lattice, pred_pos, pred_z, latent_vector
    
    def encode(self, graph):
        return self.encoder(graph)

    def decode(self, latent_vector):
        return self.decoder(latent_vector)
    
    @classmethod
    def load_decoder_for_neat(cls, checkpoint_dir, max_atoms=24, max_atomic_number=118):
        """
        Carga un checkpoint del Autoencoder desde el disco, hace la cirugía de VRAM
        y devuelve ÚNICAMENTE el modelo Decoder y sus parámetros listos para inferencia.
        """
        import orbax.checkpoint as ocp
        import os
        
        print(f"\n🔍 Buscando checkpoints en: {checkpoint_dir}")
        checkpoint_manager = ocp.CheckpointManager(
            os.path.abspath(checkpoint_dir), 
            ocp.PyTreeCheckpointer()
        )
        
        best_step = checkpoint_manager.best_step()
        if best_step is None:
            raise FileNotFoundError(f"❌ No se encontró ningún checkpoint válido en {checkpoint_dir}")
            
        print(f"📥 Cargando pesos globales (Época: {best_step})...")
        restored = checkpoint_manager.restore(best_step, args=ocp.args.PyTreeRestore())
        full_params = restored['params'] if 'params' in restored else restored
        
        # ✂️ CIRUGÍA DE MEMORIA: Extraemos solo la llave del Decoder
        decoder_only_params = full_params['decoder']
        
        # Instanciamos la arquitectura pura del Decoder
        decoder_model = CrystalDecoder(
            max_atoms=max_atoms, 
            max_atomic_number=max_atomic_number
        )
        
        print("✅ ¡Decoder aislado con éxito! Encoder destruido antes de tocar la GPU.")
        
        # Retornamos el modelo y los pesos empaquetados como flax lo necesita
        return decoder_model, {'params': decoder_only_params}