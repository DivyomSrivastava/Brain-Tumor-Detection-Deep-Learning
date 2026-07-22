"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Training Pipeline

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

import time
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

# ------------------------------------------------------------
# Project Imports
# ------------------------------------------------------------

from configs.config import *

from src.data.dataloader import (
    train_loader,
    test_loader,
    CLASS_NAMES,
)

from src.models.efficientnet import BrainTumorClassifier

# ============================================================
# Set Random Seed
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
    
# ============================================================
# Initialize Model
# ============================================================

model = BrainTumorClassifier().to(DEVICE)

# ============================================================
# Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# ============================================================
# Learning Rate Scheduler
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 70)
    print("Brain Tumor Detection using Deep Learning")
    print("=" * 70)

    set_seed()

    print(f"Device       : {DEVICE}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Classes      : {CLASS_NAMES}")
    print("=" * 70)

    # --------------------------------------------------------
    # Training History
    # --------------------------------------------------------

    history = {

        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []

    }

    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    best_accuracy = 0.0

    early_stop_counter = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch()

        val_loss, val_acc = validate_one_epoch()

        # --------------------------------------------
        # Store History
        # --------------------------------------------

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # --------------------------------------------
        # Scheduler
        # --------------------------------------------

        scheduler.step(val_acc)

        # --------------------------------------------
        # Print Results
        # --------------------------------------------

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        print(
            f"Learning Rate : {optimizer.param_groups[0]['lr']:.7f}"
        )

        # --------------------------------------------
        # Save Best Model
        # --------------------------------------------

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            early_stop_counter = 0

            torch.save(

                model.state_dict(),

                "models/best_model.pth"

            )

            print("\n✅ Best Model Saved")

        else:

            early_stop_counter += 1

            print(
                f"\nNo Improvement ({early_stop_counter}/{EARLY_STOPPING_PATIENCE})"
            )

        # --------------------------------------------
        # Early Stopping
        # --------------------------------------------

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:

            print("\n🛑 Early Stopping Triggered")

            break

    # --------------------------------------------------------
    # Training Finished
    # --------------------------------------------------------

    end_time = time.time()

    total_time = end_time - start_time

    print("\n" + "=" * 70)

    print("Training Completed")

    print("=" * 70)

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    print(f"Training Time : {total_time/60:.2f} minutes")

    print("=" * 70)

    return history


# ============================================================
# Training Function
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_images = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ----------------------------------------------
        # Clear Previous Gradients
        # ----------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------
        # Forward Pass
        # ----------------------------------------------

        outputs = model(images)

        # ----------------------------------------------
        # Compute Loss
        # ----------------------------------------------

        loss = criterion(outputs, labels)

        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        loss.backward()

        # ----------------------------------------------
        # Update Weights
        # ----------------------------------------------

        optimizer.step()

        # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        running_correct += (predictions == labels).sum().item()

        total_images += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total_images

    epoch_accuracy = 100 * running_correct / total_images

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation Function
# ============================================================

def validate_one_epoch():

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_images = 0

    progress_bar = tqdm(
        test_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # ----------------------------------------------
            # Forward Pass
            # ----------------------------------------------

            outputs = model(images)

            # ----------------------------------------------
            # Compute Loss
            # ----------------------------------------------

            loss = criterion(outputs, labels)

            # ----------------------------------------------
            # Statistics
            # ----------------------------------------------

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            running_correct += (predictions == labels).sum().item()

            total_images += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    epoch_loss = running_loss / total_images

    epoch_accuracy = 100 * running_correct / total_images

    return epoch_loss, epoch_accuracy

# ============================================================
# Run Training
# ============================================================

if __name__ == "__main__":

    main()
