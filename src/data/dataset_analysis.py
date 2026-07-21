import os
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def count_images(self):
        print("\n========== DATASET SUMMARY ==========\n")

        total_images = 0

        for split in ["Training", "Testing"]:
            split_path = os.path.join(self.dataset_path, split)

            print(f"{split} Set")

            for cls in sorted(os.listdir(split_path)):
                class_path = os.path.join(split_path, cls)

                if not os.path.isdir(class_path):
                    continue

                images = [
                    img
                    for img in os.listdir(class_path)
                    if img.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                print(f"{cls:15} : {len(images)}")

                total_images += len(images)

            print()

        print(f"Total Images : {total_images}")

    def class_distribution(self):
        classes = []
        counts = []

        train_path = os.path.join(self.dataset_path, "Training")

        for cls in sorted(os.listdir(train_path)):
            class_path = os.path.join(train_path, cls)

            if not os.path.isdir(class_path):
                continue

            num = len(
                [
                    img
                    for img in os.listdir(class_path)
                    if img.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

            classes.append(cls)
            counts.append(num)

        plt.figure(figsize=(8, 5))
        plt.bar(classes, counts)

        plt.title("Training Dataset Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")

        plt.tight_layout()
        plt.show()

    def image_sizes(self):
        sizes = Counter()

        train_path = os.path.join(self.dataset_path, "Training")

        for cls in os.listdir(train_path):

            class_path = os.path.join(train_path, cls)

            if not os.path.isdir(class_path):
                continue

            for img in os.listdir(class_path):

                path = os.path.join(class_path, img)

                image = cv2.imread(path)

                if image is None:
                    continue

                h, w = image.shape[:2]

                sizes[(w, h)] += 1

        print("\nImage Resolution Summary\n")

        for size, count in sizes.items():
            print(f"{size} -> {count} images")

    def show_samples(self):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        train_path = os.path.join(self.dataset_path, "Training")

        for i, cls in enumerate(sorted(os.listdir(train_path))):

            class_path = os.path.join(train_path, cls)

            images = [
                img
                for img in os.listdir(class_path)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            sample = random.choice(images)

            image = cv2.imread(os.path.join(class_path, sample))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axes[i].imshow(image)
            axes[i].set_title(cls)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    DATASET_PATH = "dataset/brain_mri"

    analyzer = DatasetAnalyzer(DATASET_PATH)

    analyzer.count_images()

    analyzer.class_distribution()

    analyzer.image_sizes()

    analyzer.show_samples()