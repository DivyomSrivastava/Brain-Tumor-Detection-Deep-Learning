"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Grad-CAM Visualization

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================
import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)


from configs.config import DEVICE

from src.models.efficientnet import BrainTumorClassifier
from src.data.transforms import test_transform
from src.data.dataloader import CLASS_NAMES
# ============================================================
# Load Model
# ============================================================

MODEL_PATH = "models/best_model.pth"

device = torch.device(DEVICE)

model = BrainTumorClassifier(
    freeze_features=False
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# ============================================================
# Target Layer
# ============================================================

target_layers = [
    model.model.features[8]
]

# ============================================================
# Grad-CAM Function
# ============================================================

def generate_gradcam(image_path):


    # ============================================================
    # Create Output Folder
    # ============================================================

    save_dir = "reports/gradcam"

    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # Load Image
    # ============================================================

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image).astype(np.float32) / 255.0

    input_tensor = test_transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)
    # ============================================================
    # Predict Tumor
    # ============================================================

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    prediction = CLASS_NAMES[predicted.item()]
    confidence = confidence.item() * 100

    all_probabilities = probabilities.squeeze().cpu().numpy() * 100

    # ============================================================
    # Grad-CAM
    # ============================================================

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None
    )[0]

    # ============================================================
    # Overlay
    # ============================================================

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # ============================================================
    # Save Original Image
    # ============================================================

    original = (rgb_image * 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(save_dir, "original.png"),
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    )

    # ============================================================
    # Save Heatmap
    # ============================================================

    heatmap = np.uint8(255 * grayscale_cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(
        os.path.join(save_dir, "heatmap.png"),
        heatmap
    )

    # ============================================================
    # Save Overlay
    # ============================================================

    cv2.imwrite(
        os.path.join(save_dir, "overlay.png"),
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    # ============================================================
    # Create Professional Comparison Figure
    # ============================================================

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    ax[0].imshow(original)
    ax[0].set_title(
        "Original MRI",
        fontsize=14,
        fontweight="bold"
    )
    ax[0].axis("off")

    ax[1].imshow(
        cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )
    )
    ax[1].set_title(
        "Grad-CAM Heatmap",
        fontsize=14,
        fontweight="bold"
    )
    ax[1].axis("off")

    ax[2].imshow(visualization)
    ax[2].set_title(
        "Overlay",
        fontsize=14,
        fontweight="bold"
    )
    ax[2].axis("off")

    plt.tight_layout()

    comparison_path = os.path.join(
        save_dir,
        "comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    
    # ============================================================
    # Save Prediction
    # ============================================================

    prediction_data = {

        "prediction": prediction,

        "confidence": round(confidence, 2),

        "probabilities": {

            CLASS_NAMES[i]: round(float(all_probabilities[i]), 2)

            for i in range(len(CLASS_NAMES))

        }

    }

    # ============================================================
    # Save Prediction JSON
    # ============================================================

    prediction_json_path = os.path.join(
        save_dir,
        "prediction.json"
    )
    

    with open(
        prediction_json_path,
        "w"
    ) as f:

        json.dump(
            prediction_data,
            f,
            indent=4
        )

# ============================================================
# Save Human Readable Report
# ============================================================

    prediction_txt_path = os.path.join(
        save_dir,
        "prediction.txt"
    )

    with open(
        prediction_txt_path,
        "w"
    ) as f:

        f.write("Brain Tumor Prediction Report\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Predicted Tumor : {prediction}\n")
        f.write(f"Confidence      : {confidence:.2f}%\n\n")

        f.write("Class Probabilities\n")
        f.write("-" * 45 + "\n")

        for i in range(len(CLASS_NAMES)):

            f.write(
                f"{CLASS_NAMES[i]:15s}: "
                f"{all_probabilities[i]:6.2f}%\n"
            )

        f.write("\n")
        f.write("=" * 45 + "\n")
        f.write("Generated by Brain Tumor Detection using Deep Learning\n")
    # ============================================================
    # Success Message
    # ============================================================

    print("\n========================================================")
    print("Brain Tumor Prediction")
    print("========================================================")

    print(f"Predicted Tumor : {prediction}")
    print(f"Confidence      : {confidence:.2f}%")

    print("\nClass Probabilities")

    print("--------------------------------------------------------")

    for i in range(len(CLASS_NAMES)):

        print(f"{CLASS_NAMES[i]:15s}: {all_probabilities[i]:6.2f}%")

    print("\n========================================================")

    print("Explainable AI Visualization Generated")

    print("========================================================")

    print(f"Saved in : {save_dir}")

    print("--------------------------------------------------------")

    print("original.png")

    print("heatmap.png")

    print("overlay.png")

    print("comparison.png")

    print("prediction.json")

    print("========================================================")

    return {

        "prediction": prediction,

        "confidence": confidence,

        "probabilities": prediction_data["probabilities"],

        "original": os.path.join(save_dir, "original.png"),

        "heatmap": os.path.join(save_dir, "heatmap.png"),

        "overlay": os.path.join(save_dir, "overlay.png"),

        "comparison": comparison_path,

        "prediction_json": prediction_json_path,
        "prediction_txt": prediction_txt_path
    }

# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter MRI image path : "
    ).strip()

    result = generate_gradcam(image_path)

    print("\n" + "=" * 60)
    print("Explainable AI Module Completed Successfully")
    print("=" * 60)

    print("\nGenerated Files:")

    print(f"• Original MRI      : {result['original']}")
    print(f"• Grad-CAM Heatmap  : {result['heatmap']}")
    print(f"• Overlay Image     : {result['overlay']}")
    print(f"• Comparison Image  : {result['comparison']}")
    print(f"• Prediction JSON   : {result['prediction_json']}")
    print(f"• Prediction Report : {result['prediction_txt']}")

    print("\nReturned Information:")

    print(f"Prediction          : {result['prediction']}")
    print(f"Confidence          : {result['confidence']:.2f}%")

    print("\nClass Probabilities:")

    for tumor, prob in result["probabilities"].items():
        print(f"{tumor:15s}: {prob:6.2f}%")

    print("\n" + "=" * 60)