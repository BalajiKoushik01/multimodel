
import os
import argparse
import sys

# Add parent directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import yaml
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CervicalDataset, LesionAwareDataset
from models.dual_encoder import DualEncoder
from utils.losses import InfoNCELoss, PrototypeLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    # Dataset & Dataloader
    # 1. Standard mock/flat datasets
    ds_colpo_std = CervicalDataset(config['dataset_path'], modality='colposcopy', transform=transform, mode='train')
    
    # 2. External datasets (check if they exist/are non-empty)
    # Note: parsing logic expects the root of the specific dataset, so we append the folder name
    annocerv_path = os.path.join(config['dataset_path'], 'annocerv_real')
    mobileodt_path = os.path.join(config['dataset_path'], 'mobileodt')
    
    ds_annocerv = CervicalDataset(annocerv_path, modality='annocerv', transform=transform, mode='train')
    ds_mobileodt = CervicalDataset(mobileodt_path, modality='mobileodt', transform=transform, mode='train')
    
    # Combine all colposcopy sources
    datasets_to_combine = []
    if len(ds_colpo_std) > 0: datasets_to_combine.append(ds_colpo_std)
    if len(ds_annocerv) > 0: datasets_to_combine.append(ds_annocerv)
    if len(ds_mobileodt) > 0: datasets_to_combine.append(ds_mobileodt)
    
    if not datasets_to_combine:
        print("Error: No colposcopy images found in data/colposcopy, data/annocerv, or data/mobileodt.")
        return

    from data.dataset import CombinedDataset
    dataset_colpo = CombinedDataset(datasets_to_combine)
    
    print(f"Total training images (Colposcopy): {len(dataset_colpo)}")
    print(f"  - Mock/Std: {len(ds_colpo_std)}")
    print(f"  - AnnoCerv: {len(ds_annocerv)}")
    print(f"  - MobileODT: {len(ds_mobileodt)}")

    # Histopathology (mock/std typically, unless we get external source)
    dataset_histo = CervicalDataset(config['dataset_path'], modality='histopathology', transform=transform, mode='train')
    
    # Validation check
    if len(dataset_histo) == 0:
         print("Warning: No histopathology images found. SSL will train with limited efficacy or fail if batching relies on it.")
         # In a real scenario, we might want to duplicate colpo if no histo exists just to run code, 
         # but properly we need histo data. For now, continue but warn.

    # Cycle histo info if smaller, or zip until max
    loader_colpo = DataLoader(dataset_colpo, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    loader_histo = DataLoader(dataset_histo, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    
    max_len = max(len(loader_colpo), len(loader_histo))
    print(f"Training for {max_len} batches per epoch (cycling smaller dataset).")
    
    # Create iterators
    from itertools import cycle
    if len(loader_colpo) < max_len: loader_colpo = cycle(loader_colpo)
    if len(loader_histo) < max_len: loader_histo = cycle(loader_histo)
    
    # Model
    model = DualEncoder(config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Losses
    criterion = InfoNCELoss(temperature=config['temperature']).to(device)
    proto_loss_fn = PrototypeLoss().to(device)

    # Training Loop
    model.train()
    for epoch in range(config['max_epochs']):
        epoch_loss = 0.0
        # Since we might have cycles, we can't just zip directly without handling length
        # But if we cycle the shorter one, zip will terminate on the longer one (which is now both max_len or infinite)
        # Wait, cycle makes it infinite. zip(cycle(A), B) terminates when B ends.
        # If both are cycled (which shouldn't happen if we use max_len and only cycle the shorter), then we need islice.
        # But here logic: 
        # If A < B: A = cycle(A). max_len = len(B). zip(cycle(A), B) runs for len(B). Correct.
        
        progress_bar = tqdm(zip(loader_colpo, loader_histo), total=max_len, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        
        for batch_idx, (img_c, img_h) in enumerate(progress_bar):
            img_c = img_c.to(device)
            img_h = img_h.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            z_c, z_h, feat_c, feat_h = model.forward_ssl(img_c, img_h)
            
            # Loss
            loss_con = criterion(z_c, z_h)
            loss_proto = proto_loss_fn(z_c) + proto_loss_fn(z_h)
            
            loss = loss_con + 0.1 * loss_proto # Weighting could be configurable
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / (batch_idx + 1):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging_freq'] == 0:
             if not os.path.exists(config['weights_save_path']):
                 os.makedirs(config['weights_save_path'])
             torch.save(model.state_dict(), os.path.join(config['weights_save_path'], f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
