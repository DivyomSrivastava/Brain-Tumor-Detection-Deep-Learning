"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

EfficientNet-B0 Model

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class BrainTumorClassifier(nn.Module):
    """
    EfficientNet-B0 based Brain Tumor Classifier
    """

    def __init__(
        self,
        num_classes=4,
        dropout_rate=0.3,
        freeze_features=True,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load Pretrained EfficientNet-B0
        # ----------------------------------------------------

        self.model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # ----------------------------------------------------
        # Freeze Feature Extractor
        # ----------------------------------------------------

        if freeze_features:

            for param in self.model.features.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # Replace Classifier
        # ----------------------------------------------------

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(

            nn.Dropout(dropout_rate),

            nn.Linear(
                in_features,
                num_classes
            )

        )

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    def forward(self, x):

        return self.model(x)


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

if __name__ == "__main__":

    model = BrainTumorClassifier()

    print(model)

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    output = model(dummy_input)

    print("=" * 50)
    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)
    print("=" * 50)