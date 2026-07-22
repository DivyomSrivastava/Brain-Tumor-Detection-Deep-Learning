"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Global Configuration File

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

import torch
from pathlib import Path

# ============================================================
# Project Directories
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset" / "brain_mri"

TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

MODEL_DIR = ROOT_DIR / "models"

REPORT_DIR = ROOT_DIR / "reports"

SCREENSHOT_DIR = ROOT_DIR / "screenshots"

NOTEBOOK_DIR = ROOT_DIR / "notebooks"

# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "EfficientNet-B0"

NUM_CLASSES = 4

IMAGE_SIZE = 224

DROPOUT_RATE = 0.30

FREEZE_FEATURES = True

# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 25

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_WORKERS = 0

PIN_MEMORY = True

# ============================================================
# Training Hyperparameters
# ============================================================

MODEL_NAME = "EfficientNet-B0"

NUM_EPOCHS = 25

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

SEED = 42

# ============================================================
# Device Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# Random Seed
# ============================================================

SEED = 42

# ============================================================
# Model Saving
# ============================================================

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"

# ============================================================
# Reports
# ============================================================

LOSS_CURVE = REPORT_DIR / "loss_curve.png"

ACCURACY_CURVE = REPORT_DIR / "accuracy_curve.png"

CONFUSION_MATRIX = REPORT_DIR / "confusion_matrix.png"

CLASSIFICATION_REPORT = REPORT_DIR / "classification_report.txt"

TRAINING_LOG = REPORT_DIR / "training_log.csv"

# ============================================================
# Streamlit / Prediction
# ============================================================

UPLOAD_FOLDER = ROOT_DIR / "uploads"

PDF_REPORT = REPORT_DIR / "prediction_report.pdf"

# ============================================================
# Class Names
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

# ============================================================
# Create Required Directories Automatically
# ============================================================

MODEL_DIR.mkdir(exist_ok=True)

REPORT_DIR.mkdir(exist_ok=True)

UPLOAD_FOLDER.mkdir(exist_ok=True)