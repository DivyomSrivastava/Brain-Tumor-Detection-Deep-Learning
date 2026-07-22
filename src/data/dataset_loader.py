"""
------------------------------------------------------------
Brain Tumor Detection using Deep Learning

Custom PyTorch Dataset

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""

import os

from PIL import Image
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    """
    Custom Dataset for Brain Tumor MRI Images.
    """

    def __init__(self, root_dir, transform=None):
        """
        Parameters
        ----------
        root_dir : str
            Path to dataset folder.

        transform : torchvision.transforms
            Image transformations.
        """

        self.root_dir = root_dir
        self.transform = transform

        # ----------------------------------------------------
        # Class Names
        # ----------------------------------------------------

        self.classes = sorted(
            [
                folder
                for folder in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, folder))
            ]
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        # ----------------------------------------------------
        # Store image paths and labels
        # ----------------------------------------------------

        self.samples = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )

        for class_name in self.classes:

            class_folder = os.path.join(root_dir, class_name)

            for image_name in os.listdir(class_folder):

                if image_name.lower().endswith(valid_extensions):

                    image_path = os.path.join(
                        class_folder,
                        image_name,
                    )

                    label = self.class_to_idx[class_name]

                    self.samples.append(
                        (image_path, label)
                    )

    # --------------------------------------------------------
    # Number of Images
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    # Return one image
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label