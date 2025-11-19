"""
Quick test script to verify EVA-Net can run
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models.eva_net import EVANet
from dataset import EEGBrainAgeDataset
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Composite loss: L_total = L_pred + β * L_IB + γ * L_align"""
    def __init__(self, beta=1e-3, gamma=0.7):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, age_pred, age_true, z, prototype, kl_loss):
        age_pred = age_pred.squeeze(-1)
        L_pred = self.mse(age_pred, age_true)
        L_IB = kl_loss
        L_align = torch.mean((z - prototype) ** 2)
        total_loss = L_pred + self.beta * L_IB + self.gamma * L_align

        loss_dict = {
            'total': total_loss.item(),
            'pred': L_pred.item(),
            'ib': L_IB.item(),
            'align': L_align.item()
        }
        return total_loss, loss_dict

print("=" * 80)
print("EVA-Net Quick Test")
print("=" * 80)

# Set device
device = torch.device('cpu')
print(f"\nDevice: {device}")

# Load a small subset of data
print("\nLoading dataset...")
try:
    train_dataset = EEGBrainAgeDataset("preprocessed_data", exclude_pathological=True)
    print(f"✓ Dataset loaded: {len(train_dataset)} epochs")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# Create dataloader with small batch
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# Create model
print("\nInitializing model...")
try:
    model = EVANet(
        n_channels=19,
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        sampling_factor=5,
        latent_dim=64,
        hidden_dim=256
    ).to(device)
    print(f"✓ Model initialized")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    exit(1)

# Test forward pass
print("\nTesting forward pass...")
try:
    batch = next(iter(train_loader))
    eeg, age = batch
    eeg = eeg.to(device)
    age = age.to(device)

    print(f"  Input shape: {eeg.shape}")
    print(f"  Age shape: {age.shape}")

    with torch.no_grad():
        age_pred, z, prototype, kl_loss = model(eeg, age)

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {age_pred.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Prototype shape: {prototype.shape}")
    print(f"  KL loss: {kl_loss.item():.4f}")
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test loss computation
print("\nTesting loss computation...")
try:
    criterion = CompositeLoss(beta=1e-3, gamma=0.7)
    age_pred, z, prototype, kl_loss = model(eeg, age)
    loss, loss_dict = criterion(age_pred, age, z, prototype, kl_loss)

    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Pred loss: {loss_dict['pred']:.4f}")
    print(f"  IB loss: {loss_dict['ib']:.4f}")
    print(f"  Align loss: {loss_dict['align']:.4f}")
except Exception as e:
    print(f"✗ Error computing loss: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test backward pass
print("\nTesting backward pass...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Backward pass successful")
except Exception as e:
    print(f"✗ Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test one mini training epoch
print("\nTesting mini training (3 batches)...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CompositeLoss(beta=1e-3, gamma=0.7)

    for batch_idx, (eeg, age) in enumerate(train_loader):
        if batch_idx >= 3:
            break

        eeg = eeg.to(device)
        age = age.to(device)

        age_pred, z, prototype, kl_loss = model(eeg, age)
        loss, loss_dict = criterion(age_pred, age, z, prototype, kl_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Batch {batch_idx + 1}: Loss = {loss_dict['total']:.4f}")

    print(f"✓ Mini training successful")
except Exception as e:
    print(f"✗ Error in training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed! EVA-Net is working correctly.")
print("=" * 80)
