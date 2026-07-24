"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Model Evaluation

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

import os
import json

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from configs.config import *

from src.models.efficientnet import BrainTumorClassifier
from src.data.dataloader import test_loader, CLASS_NAMES

# ============================================================
# Output Folder
# ============================================================

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Load Model
# ============================================================

model = BrainTumorClassifier().to(DEVICE)

model.load_state_dict(

    torch.load(
        "models/best_model.pth",
        map_location=DEVICE,
    )

)

model.eval()

# ============================================================
# Evaluation
# ============================================================

def evaluate_model():

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return np.array(all_labels), np.array(all_predictions)

# ============================================================
# Compute Metrics
# ============================================================

def compute_metrics(labels, predictions):

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
    )

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        report,
        matrix,
    )

# ============================================================
# Save Results
# ============================================================

def save_results(
    accuracy,
    precision,
    recall,
    f1,
    report,
    matrix,
):

    metrics = {

        "Accuracy": float(accuracy),

        "Precision": float(precision),

        "Recall": float(recall),

        "F1 Score": float(f1),

    }

    with open(
        os.path.join(REPORT_DIR, "metrics.json"),
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    with open(
        os.path.join(
            REPORT_DIR,
            "classification_report.txt",
        ),
        "w",
    ) as file:

        file.write(report)

    np.save(
        os.path.join(
            REPORT_DIR,
            "confusion_matrix.npy",
        ),
        matrix,
    )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Evaluating Best Model")
    print("=" * 60)

    labels, predictions = evaluate_model()

    (
        accuracy,
        precision,
        recall,
        f1,
        report,
        matrix,
    ) = compute_metrics(
        labels,
        predictions,
    )

    save_results(
        accuracy,
        precision,
        recall,
        f1,
        report,
        matrix,
    )

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")
    print(report)

    print("\nConfusion Matrix\n")
    print(matrix)

    print("\nEvaluation Completed.")

    print(f"\nResults Saved in '{REPORT_DIR}'")