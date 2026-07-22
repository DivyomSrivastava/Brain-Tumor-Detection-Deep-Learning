"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

PyTorch DataLoader

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

from torch.utils.data import DataLoader

from src.data.dataset_loader import BrainTumorDataset
from src.data.transforms import train_transform, test_transform

# ------------------------------------------------------------
# Dataset Paths
# ------------------------------------------------------------

TRAIN_DIR = "dataset/brain_mri/Training"
TEST_DIR = "dataset/brain_mri/Testing"

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0

# ------------------------------------------------------------
# Training Dataset
# ------------------------------------------------------------

train_dataset = BrainTumorDataset(
    root_dir=TRAIN_DIR,
    transform=train_transform
)

# ------------------------------------------------------------
# Testing Dataset
# ------------------------------------------------------------

test_dataset = BrainTumorDataset(
    root_dir=TEST_DIR,
    transform=test_transform
)

# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ------------------------------------------------------------
# Class Information
# ------------------------------------------------------------

CLASS_NAMES = train_dataset.classes

if __name__ == "__main__":

    print("=" * 50)
    print("Brain Tumor Dataset")
    print("=" * 50)

    print(f"Training Images : {len(train_dataset)}")
    print(f"Testing Images  : {len(test_dataset)}")
    print(f"Classes         : {CLASS_NAMES}")

    print("=" * 50)

    images, labels = next(iter(train_loader))

    print(f"Batch Shape     : {images.shape}")
    print(f"Labels Shape    : {labels.shape}")

    print("=" * 50)