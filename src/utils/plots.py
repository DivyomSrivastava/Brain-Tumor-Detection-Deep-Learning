"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Training & Evaluation Plots

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# Paths
# ============================================================

REPORT_DIR = "reports"

HISTORY_FILE = os.path.join(REPORT_DIR, "history.json")

CONFUSION_MATRIX_FILE = os.path.join(
    REPORT_DIR,
    "confusion_matrix.npy"
)

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

# ============================================================
# Create Report Folder
# ============================================================

os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Accuracy & Loss Curves
# ============================================================

if os.path.exists(HISTORY_FILE):

    print("History file found.")

    with open(HISTORY_FILE, "r") as f:

        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = range(1, len(train_loss) + 1)

    # --------------------------------------------------------
    # Accuracy Curve
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.plot(
        epochs,
        train_acc,
        marker="o",
        linewidth=2,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        val_acc,
        marker="s",
        linewidth=2,
        label="Validation Accuracy",
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "accuracy_curve.png"
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Loss Curve
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        linewidth=2,
        label="Training Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        marker="s",
        linewidth=2,
        label="Validation Loss",
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "loss_curve.png"
        ),
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Combined Summary
    # --------------------------------------------------------

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    plt.plot(epochs, train_acc, label="Train")

    plt.plot(epochs, val_acc, label="Validation")

    plt.title("Accuracy")

    plt.grid(True)

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(epochs, train_loss, label="Train")

    plt.plot(epochs, val_loss, label="Validation")

    plt.title("Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "training_summary.png"
        ),
        dpi=300,
    )

    plt.close()

else:

    print("history.json not found.")

    print("Skipping Accuracy/Loss Plots.")

# ============================================================
# Confusion Matrix
# ============================================================

if os.path.exists(CONFUSION_MATRIX_FILE):

    print("Confusion matrix found.")

    matrix = np.load(CONFUSION_MATRIX_FILE)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "confusion_matrix.png"
        ),
        dpi=300,
    )

    plt.close()

else:

    print("confusion_matrix.npy not found.")

    print("Skipping Confusion Matrix.")

# ============================================================
# Finished
# ============================================================

print("\n" + "="*60)

print("Plot Generation Finished")

print("="*60)

print("Generated files are available in:")

print(REPORT_DIR)

print("="*60)