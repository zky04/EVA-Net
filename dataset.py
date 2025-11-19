"""
Dataset and DataLoader for EVA-Net
Loads preprocessed EEG data from .npz files, filtering out pathological data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class EEGBrainAgeDataset(Dataset):
    """
    Dataset for loading preprocessed EEG epochs with age labels.
    Each .npz file contains multiple 4-second epochs from one subject.
    """

    def __init__(self, data_dir, exclude_pathological=True, expected_channels=19):
        """
        Args:
            data_dir: Path to directory containing .npz files (train or test)
            exclude_pathological: If True, filters out files with pathological indicators
            expected_channels: Expected number of EEG channels (default: 19)
        """
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob("*.npz")))
        self.expected_channels = expected_channels

        # Filter out pathological data if needed
        if exclude_pathological:
            self.file_list = self._filter_pathological(self.file_list)

        # Prepare index mapping: (file_idx, epoch_idx)
        self.index_mapping = []
        self.ages = []
        skipped_files = 0

        print(f"Loading dataset from {data_dir}...")
        for file_idx, file_path in enumerate(self.file_list):
            if file_idx % 100 == 0 and file_idx > 0:
                print(f"  Processed {file_idx}/{len(self.file_list)} files...")

            data = np.load(file_path)
            epochs_data = data['epochs_data']  # [n_epochs, channels, time]

            # Skip files with incorrect number of channels
            if epochs_data.shape[1] != expected_channels:
                skipped_files += 1
                continue

            age = float(data['age'])
            n_epochs = epochs_data.shape[0]

            for epoch_idx in range(n_epochs):
                self.index_mapping.append((file_idx, epoch_idx))
                self.ages.append(age)

        self.ages = np.array(self.ages)
        print(f"Loaded {len(self.index_mapping)} epochs from {len(self.file_list) - skipped_files} subjects")
        if skipped_files > 0:
            print(f"  Skipped {skipped_files} files with incorrect channel count")
        print(f"Age range: {self.ages.min():.1f} - {self.ages.max():.1f} years")
        print(f"Mean age: {self.ages.mean():.1f} years")

    def _filter_pathological(self, file_list):
        """
        Filter out files that may contain pathological data.
        This is a placeholder - adjust based on your file naming convention.
        """
        # According to paper, we only use healthy subjects for training
        # Assuming pathological files might have certain naming patterns
        # For now, we keep all files in train/test folders as they should be pre-filtered
        return file_list

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_idx, epoch_idx = self.index_mapping[idx]
        file_path = self.file_list[file_idx]

        # Load data
        data = np.load(file_path)
        epochs_data = data['epochs_data']  # [n_epochs, 19, 1001]
        age = float(data['age'])

        # Get specific epoch: [19, 1001]
        epoch = epochs_data[epoch_idx]

        # Trim to exactly 1000 time points (4 seconds at 250 Hz)
        # Paper specifies: [19 channels, 1000 time points]
        epoch = epoch[:, :1000]  # [19, 1000]

        # Convert to torch tensors
        epoch = torch.from_numpy(epoch).float()
        age = torch.tensor(age, dtype=torch.float32)

        return epoch, age


class SubjectEEGDataset(Dataset):
    """
    Dataset that groups all epochs by subject.
    Useful for subject-level evaluation and cross-validation.
    """

    def __init__(self, data_dir, exclude_pathological=True):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob("*.npz")))

        if exclude_pathological:
            self.file_list = self._filter_pathological(self.file_list)

        print(f"Loading subject-level dataset from {data_dir}...")
        print(f"Found {len(self.file_list)} subjects")

    def _filter_pathological(self, file_list):
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        epochs_data = data['epochs_data'][:, :, :1000]  # [n_epochs, 19, 1000]
        age = float(data['age'])

        epochs_data = torch.from_numpy(epochs_data).float()
        age = torch.tensor(age, dtype=torch.float32)

        return epochs_data, age, file_path.stem


def create_10fold_splits(data_dir, seed=42):
    """
    Create 10-fold cross-validation splits at the subject level.

    Args:
        data_dir: Path to directory containing all subject .npz files
        seed: Random seed for reproducibility

    Returns:
        folds: List of 10 tuples, each containing (train_files, test_files)
    """
    np.random.seed(seed)
    data_dir = Path(data_dir)
    all_files = sorted(list(data_dir.glob("*.npz")))

    # Shuffle files
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)
    all_files = [all_files[i] for i in indices]

    # Create 10 folds
    n_files = len(all_files)
    fold_size = n_files // 10
    folds = []

    for fold_idx in range(10):
        # Define test set for this fold
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < 9 else n_files
        test_files = all_files[test_start:test_end]

        # Remaining files are for training
        train_files = all_files[:test_start] + all_files[test_end:]

        folds.append((train_files, test_files))

        print(f"Fold {fold_idx + 1}: {len(train_files)} train, {len(test_files)} test subjects")

    return folds


def get_dataloader(file_list, batch_size=64, shuffle=True, num_workers=4):
    """
    Create a DataLoader from a list of .npz files.

    Args:
        file_list: List of Path objects pointing to .npz files
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    # Create temporary directory with symlinks (or use file list directly)
    # For simplicity, we create a custom dataset from file list
    dataset = EEGDatasetFromFiles(file_list)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class EEGDatasetFromFiles(Dataset):
    """Dataset that loads from a specific list of files"""

    def __init__(self, file_list):
        self.file_list = file_list
        self.index_mapping = []
        self.ages = []

        for file_idx, file_path in enumerate(file_list):
            data = np.load(file_path)
            epochs_data = data['epochs_data']
            age = float(data['age'])

            n_epochs = epochs_data.shape[0]
            for epoch_idx in range(n_epochs):
                self.index_mapping.append((file_idx, epoch_idx))
                self.ages.append(age)

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_idx, epoch_idx = self.index_mapping[idx]
        file_path = self.file_list[file_idx]

        data = np.load(file_path)
        epochs_data = data['epochs_data']
        age = float(data['age'])

        epoch = epochs_data[epoch_idx, :, :1000]  # [19, 1000]

        epoch = torch.from_numpy(epoch).float()
        age = torch.tensor(age, dtype=torch.float32)

        return epoch, age


if __name__ == "__main__":
    # Test dataset loading
    train_dataset = EEGBrainAgeDataset("preprocessed_data/train", exclude_pathological=True)
    test_dataset = EEGBrainAgeDataset("preprocessed_data/test", exclude_pathological=True)

    print(f"\nTrain dataset: {len(train_dataset)} epochs")
    print(f"Test dataset: {len(test_dataset)} epochs")

    # Test dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    for batch_idx, (eeg, age) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  EEG shape: {eeg.shape}")  # [64, 19, 1000]
        print(f"  Age shape: {age.shape}")  # [64]
        print(f"  Age range: {age.min():.1f} - {age.max():.1f}")
        break
