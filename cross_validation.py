"""
10-fold cross-validation for EVA-Net
Subject-level stratified splits to prevent data leakage.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from models.eva_net import EVANet
from dataset import create_10fold_splits, EEGDatasetFromFiles


class CompositeLoss(nn.Module):
    """
    Composite loss for EVA-Net training:
    L_total = L_pred + β * L_IB + γ * L_align
    """

    def __init__(self, beta=1e-3, gamma=0.7):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, age_pred, age_true, z, prototype, kl_loss):
        """
        Args:
            age_pred: [batch_size, 1] - predicted ages
            age_true: [batch_size] - true chronological ages
            z: [batch_size, latent_dim] - latent representations
            prototype: [batch_size, latent_dim] - age-conditioned prototypes
            kl_loss: scalar - KL divergence from VIB

        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual loss components
        """
        # L_pred: MSE between predicted and true age
        age_pred = age_pred.squeeze(-1)
        L_pred = self.mse(age_pred, age_true)

        # L_IB: KL divergence
        L_IB = kl_loss

        # L_align: MSE between latent representation and prototype
        L_align = torch.mean((z - prototype) ** 2)

        # Total composite loss
        total_loss = L_pred + self.beta * L_IB + self.gamma * L_align

        loss_dict = {
            'total': total_loss.item(),
            'pred': L_pred.item(),
            'ib': L_IB.item(),
            'align': L_align.item()
        }

        return total_loss, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_losses = {'total': 0, 'pred': 0, 'ib': 0, 'align': 0}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (eeg, age) in enumerate(pbar):
        eeg = eeg.to(device)
        age = age.to(device)

        # Forward pass
        age_pred, z, prototype, kl_loss = model(eeg, age)

        # Compute composite loss
        loss, loss_dict = criterion(age_pred, age, z, prototype, kl_loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'pred': f"{loss_dict['pred']:.4f}",
            'align': f"{loss_dict['align']:.4f}"
        })

    # Average losses
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    return avg_losses


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_losses = {'total': 0, 'pred': 0, 'ib': 0, 'align': 0}
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for eeg, age in tqdm(dataloader, desc="Validating"):
            eeg = eeg.to(device)
            age = age.to(device)

            # Forward pass
            age_pred, z, prototype, kl_loss = model(eeg, age)

            # Compute loss
            loss, loss_dict = criterion(age_pred, age, z, prototype, kl_loss)

            # Accumulate
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1

            all_preds.append(age_pred.squeeze(-1).cpu().numpy())
            all_targets.append(age.cpu().numpy())

    # Average losses
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Concatenate predictions
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute metrics
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    return avg_losses, metrics


def run_cross_validation(data_dir, config, save_dir):
    """
    Run 10-fold cross-validation at subject level.

    Args:
        data_dir: Path to directory containing all preprocessed .npz files
        config: Configuration dictionary
        save_dir: Directory to save results

    Returns:
        results: Dictionary with results from all folds
    """
    # Create 10-fold splits
    print("Creating 10-fold cross-validation splits...")
    folds = create_10fold_splits(data_dir, seed=config.get('seed', 42))

    # Store results from each fold
    all_fold_results = []

    for fold_idx, (train_files, val_files) in enumerate(folds):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}/10")
        print("=" * 80)

        # Train on this fold
        fold_metrics = train_single_fold(
            train_files=train_files,
            val_files=val_files,
            config=config,
            fold_idx=fold_idx,
            save_dir=save_dir
        )

        all_fold_results.append(fold_metrics)

        # Save intermediate results
        results_dict = {
            'fold_results': all_fold_results,
            'config': config
        }
        with open(save_dir / 'cv_results_partial.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

    # Compute statistics across folds
    mae_values = [r['mae'] for r in all_fold_results]
    rmse_values = [r['rmse'] for r in all_fold_results]
    r2_values = [r['r2'] for r in all_fold_results]

    final_results = {
        'fold_results': all_fold_results,
        'summary': {
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values)
        },
        'config': config
    }

    # Print summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nResults averaged over 10 folds:")
    print(f"  MAE:  {final_results['summary']['mae_mean']:.3f} ± {final_results['summary']['mae_std']:.3f}")
    print(f"  RMSE: {final_results['summary']['rmse_mean']:.3f} ± {final_results['summary']['rmse_std']:.3f}")
    print(f"  R²:   {final_results['summary']['r2_mean']:.3f} ± {final_results['summary']['r2_std']:.3f}")

    # Save final results
    with open(save_dir / 'cv_results_final.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    return final_results


def train_single_fold(train_files, val_files, config, fold_idx, save_dir):
    """
    Train model on a single fold.

    Args:
        train_files: List of training file paths
        val_files: List of validation file paths
        config: Configuration dictionary
        fold_idx: Fold index (for saving)
        save_dir: Directory to save checkpoints

    Returns:
        best_metrics: Dictionary with best validation metrics
    """
    device = torch.device(config['device'])

    # Create dataloaders
    train_dataset = EEGDatasetFromFiles(train_files)
    val_dataset = EEGDatasetFromFiles(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train: {len(train_dataset)} epochs from {len(train_files)} subjects")
    print(f"  Val: {len(val_dataset)} epochs from {len(val_files)} subjects")

    # Create model
    model = EVANet(
        n_channels=config['n_channels'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        sampling_factor=config['sampling_factor'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)

    # Loss and optimizer
    criterion = CompositeLoss(beta=config['beta'], gamma=config['gamma'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] / 10
    )

    # Training loop
    best_mae = float('inf')
    best_metrics = None
    patience_counter = 0
    training_history = []

    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_losses, val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history_entry = {
            'epoch': epoch,
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2']
        }
        training_history.append(history_entry)

        # Print epoch summary
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{config['epochs']}:")
            print(f"  Train Loss: {train_losses['total']:.4f} (pred: {train_losses['pred']:.4f}, "
                  f"ib: {train_losses['ib']:.4f}, align: {train_losses['align']:.4f})")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.3f}, RMSE: {val_metrics['rmse']:.3f}, R²: {val_metrics['r2']:.3f}")

        # Save best model
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            best_metrics = val_metrics.copy()
            best_metrics['best_epoch'] = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = save_dir / f"fold_{fold_idx + 1}_best.pth"
            torch.save({
                'epoch': epoch,
                'fold_idx': fold_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config,
                'training_history': training_history
            }, checkpoint_path)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  → Saved best model (MAE: {best_mae:.3f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print(f"\nFold {fold_idx + 1} complete. Best MAE: {best_mae:.3f} at epoch {best_metrics['best_epoch']}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="10-fold cross-validation for EVA-Net")
    parser.add_argument('--data_dir', type=str, default='preprocessed_data',
                        help='Directory containing all preprocessed subject files')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cv_results',
                        help='Directory to save CV results')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configuration
    config = {
        # Model architecture
        'n_channels': 19,
        'd_model': 128,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 512,
        'dropout': 0.1,
        'sampling_factor': 5,
        'latent_dim': 64,
        'hidden_dim': 256,

        # Loss weights
        'beta': 1e-3,  # VIB weight
        'gamma': 0.7,  # Alignment weight

        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'patience': 20,

        # System
        'device': args.device,
        'num_workers': args.num_workers,
        'seed': args.seed
    }

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("EVA-Net 10-Fold Cross-Validation")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nData directory: {args.data_dir}")
    print(f"Results will be saved to: {save_dir}")

    # Run cross-validation
    start_time = datetime.now()
    results = run_cross_validation(args.data_dir, config, save_dir)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds() / 3600
    print(f"\n✓ Cross-validation complete! Total time: {duration:.2f} hours")
    print(f"Results saved to: {save_dir / 'cv_results_final.json'}")


if __name__ == "__main__":
    main()
