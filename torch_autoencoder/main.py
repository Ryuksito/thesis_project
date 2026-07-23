# autoencoder/main.py
import os
import sys
import time
import csv
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

# Resolución estricta de rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from torch_autoencoder.config import *
import config as gconfig
from torch_autoencoder.data.loader import CrystalDataset
from torch_autoencoder.models.autoencoder import Autoencoder
from torch_autoencoder.loss import compute_total_loss

# =====================================================================
# GESTIÓN DE DISPOSITIVOS Y VRAM NATIVA
# =====================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_peak_vram(device):
    """Obtiene el pico de uso de VRAM de la época actual usando la API nativa."""
    if device.type == 'cuda':
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
        return f"{vram_mb:.0f} MB"
    return "N/A"

# =====================================================================
# BUCLE PRINCIPAL
# =====================================================================
def main():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python import paths: {sys.path[-1]}")
    
    LOGS_DIR = os.path.join(BASE_DIR, "torch_autoencoder", "runs", "v1")
    print(f"Los logs se guardarán en: {LOGS_DIR}")

    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO DEL AUTOENCODER CRISTALOGRÁFICO")
    print("="*60)
    
    device = get_device()
    print(f"🖥️ Acelerador detectado: {device}")

    # --- PREPARACIÓN DE DIRECTORIOS ---
    os.makedirs(LOGS_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(LOGS_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    csv_log_path = os.path.join(LOGS_DIR, "training_logs.csv")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    # --- DATASETS Y DATALOADERS ---
    dataset = CrystalDataset(
        h5_path=gconfig.DATA_PATH + "/autoencoder_16k.h5",
        split='train',
        max_lattice_length=gconfig.MAX_LATTICE_LENGTH,
        max_lattice_angle=gconfig.MAX_LATTICE_ANGLE,
        cutoff=5.0
    )
    
    # 🔥 CORRECCIÓN: Adaptación dinámica de workers y memoria según la arquitectura
    is_cuda = (device.type == 'cuda')
    num_workers = 4 if is_cuda else 0
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=num_workers, 
        pin_memory=is_cuda, 
        persistent_workers=(num_workers > 0)
    )
    
    TOTAL_STEPS = len(dataloader) * EPOCHS

    # --- INICIALIZACIÓN DEL MODELO ---
    model = Autoencoder(
        latent_dim=gconfig.LATENT_DIM, 
        max_atoms=gconfig.MAX_ATOMS, 
        max_atomic_number=gconfig.MAX_ATOMIC_NUMBER
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR_MAX)
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=TOTAL_STEPS, 
        eta_min=LR_MIN
    )

    # --- RESTAURAR CHECKPOINT ---
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Checkpoint encontrado en {checkpoint_path}. Restaurando...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("📊 El historial CSV continuará escribiéndose.")
    else:
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss", "lattice_loss", "position_loss", 
                             "z_loss", "learning_rate", "vram_mb", "epoch_time_sec"])

    # --- ENTRENAMIENTO ---
    print(f"\n🔥 ¡Arrancando! Entrenando hasta la época {EPOCHS}...\n")
    best_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        model.train()
        
        epoch_step_metrics = []
        batch_losses = [] 
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_lattice, pred_pos, pred_z_logits, _ = model(batch)
            
            target_z_dense, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=gconfig.MAX_ATOMS)
            target_pos_dense, _ = to_dense_batch(batch.pos, batch.batch, max_num_nodes=gconfig.MAX_ATOMS)
            target_lattice = batch.norm_lattice.view(-1, 6)
            
            # Cálculo de pérdida (Los retornos ahora son TENSORES en la GPU)
            loss, (l_lat, l_z, l_pos) = compute_total_loss(
                pred_lattice, pred_pos, pred_z_logits, 
                target_lattice, target_pos_dense, target_z_dense
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 🔥 CORRECCIÓN: Extracción de métricas espaciada (Evita el lag CPU-GPU)
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                loss_val = loss.item()
                l_lat_val = l_lat.item()
                l_pos_val = l_pos.item()
                l_z_val = l_z.item()
                
                epoch_step_metrics.append([
                    epoch + 1, step + 1, loss_val, l_lat_val, l_pos_val, l_z_val, current_lr
                ])
                
                batch_losses.append(loss_val)
                pbar.set_postfix({'Loss': f"{sum(batch_losses[-50:]) / min(len(batch_losses), 50):.4f}"})
                
        # --- FIN DE LA ÉPOCA ---
        epoch_time = time.time() - start_time
        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float('inf')
        current_vram = get_peak_vram(device)
        
        # Escritura en CSV
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in epoch_step_metrics:
                row.extend([current_vram, f"{epoch_time:.2f}"])
            writer.writerows(epoch_step_metrics)

        # Guardado de Checkpoint basado en el mejor loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
        
        print(f"✅ Epoch {epoch+1} | Loss Avg: {avg_loss:.4f} | VRAM: {current_vram} | Logs CSV guardados.")

    print("\n🎉 ENTRENAMIENTO FINALIZADO. Pesos guardados con éxito en formato nativo de PyTorch.")

if __name__ == "__main__":
    main()