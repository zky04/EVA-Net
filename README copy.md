# EVA-Net: Brain Age Prediction from EEG

Efficient Variational Alignment Network for brain age estimation using EEG signals.

## Features

- **ProbSparse Attention**: O(L log L) complexity for long-sequence processing
- **Variational Information Bottleneck**: Robust latent representations
- **Continuous Prototype Network**: Age-conditioned prototype alignment
- **Multi-task Learning**: Combined prediction and alignment objectives

## Quick Start

### Requirements

```bash
pip install torch numpy scipy scikit-learn tqdm
```

### Data Preparation

Place all preprocessed EEG `.npz` files in a single directory (e.g., `preprocessed_data/`). Each file should contain:
- `epochs`: shape `[n_epochs, n_channels, time_points]` (e.g., `[N, 19, 1000]`)
- `ages`: shape `[n_epochs]`
- `is_pathological`: shape `[n_epochs]` (0=healthy, 1=pathological)

The 10-fold cross-validation will automatically split subjects into folds.

### Quick Test

Verify installation and model functionality:

```bash
python quick_test.py
```

### 10-Fold Cross-Validation

Train with subject-level stratified 10-fold CV:

```bash
python cross_validation.py \
    --data_dir preprocessed_data \
    --save_dir checkpoints/cv_results \
    --epochs 200 \
    --batch_size 64 \
    --device cuda
```

**Parameters:**
- `--data_dir`: Directory containing preprocessed `.npz` files
- `--save_dir`: Output directory for checkpoints and results
- `--epochs`: Maximum epochs per fold (default: 200)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: `cuda` or `cpu` (auto-detected)
- `--seed`: Random seed (default: 42)

### Model Configuration

Default architecture (1M parameters):

```python
n_channels: 19          # EEG channels
d_model: 128            # Transformer dimension
n_layers: 4             # Encoder layers
n_heads: 8              # Attention heads
d_ff: 512               # FFN hidden size
latent_dim: 64          # VIB latent dimension
hidden_dim: 256         # MLP hidden size
```

Loss function weights:
```python
beta: 1e-3              # VIB regularization
gamma: 0.7              # Prototype alignment
```

## Project Structure

```
.
├── models/
│   └── eva_net.py              # Model architecture
├── dataset.py                   # Data loading utilities
├── quick_test.py               # Quick functionality test
├── cross_validation.py         # 10-fold CV training
└── README.md                    # This file
```

## Output

Cross-validation produces:
- `fold_N_best.pth`: Best checkpoint for each fold
- `cv_results_final.json`: Aggregated metrics (MAE, RMSE, R²)
- `config.json`: Training configuration

## License

MIT License
