"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Prediction / Inference Script

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from configs.config import *

from src.models.efficientnet import BrainTumorClassifier
from src.data.transforms import test_transform

# ============================================================
# Load Model
# ============================================================

MODEL_PATH = "models/best_model.pth"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

device = torch.device(DEVICE)

model = BrainTumorClassifier().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# ============================================================
# Prediction Function
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found : {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = test_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASS_NAMES[prediction.item()]

    confidence = confidence.item() * 100

    all_probabilities = {}

    for idx, class_name in enumerate(CLASS_NAMES):

        all_probabilities[class_name] = (
            probabilities[0][idx].item() * 100
        )

    return predicted_class, confidence, all_probabilities


# ============================================================
# Display Result
# ============================================================

def print_prediction(image_path):

    predicted_class, confidence, probs = predict_image(
        image_path
    )

    print("=" * 60)

    print("Brain Tumor Prediction")

    print("=" * 60)

    print(f"Image : {image_path}")

    print()

    print(f"Predicted Class : {predicted_class}")

    print(f"Confidence      : {confidence:.2f}%")

    print()

    print("All Class Probabilities")

    print("-" * 60)

    for class_name, probability in probs.items():

        print(
            f"{class_name:<15} : {probability:.2f}%"
        )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = input(
        "Enter MRI image path : "
    ).strip()

    print_prediction(IMAGE_PATH)