from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Change this image if needed
# -----------------------------
IMAGE_PATH = Path("data/raw/DRIVE/training/images/21_training.tif")


def extract_green_channel(image):
    """Extract the green channel from an RGB fundus image."""
    return image[:, :, 1]


def apply_clahe(gray):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize(image):
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Cannot find {IMAGE_PATH}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = extract_green_channel(image)
    clahe = apply_clahe(green)
    normalized = normalize(clahe)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(green, cmap="gray")
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(normalized, cmap="gray")
    plt.title("CLAHE + Normalized")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
