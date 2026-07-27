"""
------------------------------------------------------------
NeuroVision AI

Grad-CAM Explainable AI Module

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.predict import (
    model,
    device,
    predict_image
)

from src.data.transforms import test_transform


# ============================================================
# Target Layer
# ============================================================

target_layers = [
    model.model.features[8]
]


# ============================================================
# Generate Grad-CAM
# ============================================================

def generate_gradcam(image_path, image_name):

    # --------------------------------------------------------
    # Safe Folder Name
    # --------------------------------------------------------

    safe_name = os.path.splitext(image_name)[0]

    safe_name = safe_name.replace(" ", "_")

    safe_name = safe_name.replace("(", "")

    safe_name = safe_name.replace(")", "")

    save_dir = os.path.join(

        "reports",

        "gradcam",

        safe_name

    )

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    # --------------------------------------------------------
    # Load Image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image).astype(np.float32) / 255.0

    input_tensor = test_transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # GradCAM
    # --------------------------------------------------------

    cam = GradCAM(

        model=model,

        target_layers=target_layers

    )

    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=None

    )[0]

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    visualization = show_cam_on_image(

        rgb_image,

        grayscale_cam,

        use_rgb=True

    )

    # --------------------------------------------------------
    # Original Image
    # --------------------------------------------------------

    original = (rgb_image * 255).astype(np.uint8)

    original_path = os.path.join(

        save_dir,

        "original.png"

    )

    cv2.imwrite(

        original_path,

        cv2.cvtColor(

            original,

            cv2.COLOR_RGB2BGR

        )

    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(

        255 * grayscale_cam

    )

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )

    heatmap_path = os.path.join(

        save_dir,

        "heatmap.png"

    )

    cv2.imwrite(

        heatmap_path,

        heatmap

    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay_path = os.path.join(

        save_dir,

        "overlay.png"

    )

    cv2.imwrite(

        overlay_path,

        cv2.cvtColor(

            visualization,

            cv2.COLOR_RGB2BGR

        )

    )

    # --------------------------------------------------------
    # Comparison Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(

        1,

        3,

        figsize=(18,6)

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

    ax[2].imshow(

        visualization

    )

    ax[2].set_title(

        "AI Focus Area",

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

    # --------------------------------------------------------
    # Return Paths
    # --------------------------------------------------------

    return {

        "folder": save_dir,

        "original": original_path,

        "heatmap": heatmap_path,

        "overlay": overlay_path,

        "comparison": comparison_path

    }


# ============================================================
# Terminal Testing
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter MRI image path : "
    ).strip()

    result = generate_gradcam(
        image_path
    )

    print("\n========================================")
    print("NeuroVision AI")
    print("========================================")
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}%")
    print("========================================")

    print("\nGenerated Files\n")

    print(result["original"])
    print(result["heatmap"])
    print(result["overlay"])
    print(result["comparison"])