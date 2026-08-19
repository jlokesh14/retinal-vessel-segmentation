from pathlib import Path
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "processed" / "DRIVE" / "training" / "images"
MASK_DIR = ROOT / "data" / "raw" / "DRIVE" / "training" / "1st_manual"

OUTPUT_IMAGE_DIR = (
    ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "images"
)
OUTPUT_MASK_DIR = (
    ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "masks"
)

PATCH_SIZE = 64
STRIDE = 64
MIN_FOV_RATIO = 0.50


def extract_patches(image, mask, image_name):
    height, width = image.shape[:2]

    count = 0

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            image_patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            fov_patch = image_patch > 0
            if fov_patch.mean() < MIN_FOV_RATIO:
                continue

            mask_patch = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            patch_name = f"{image_name}_{count:04d}.png"

            cv2.imwrite(str(OUTPUT_IMAGE_DIR / patch_name), image_patch)

            cv2.imwrite(str(OUTPUT_MASK_DIR / patch_name), mask_patch)

            count += 1

    return count


def main():

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    total_patches = 0

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    print(f"Found {len(image_files)} DRIVE training images.")

    for image_file in image_files:

        mask_file = MASK_DIR / image_file.name.replace("_training.png", "_manual1.gif")

        if not mask_file.exists():
            print(f"Mask not found: {image_file.name}")
            continue

        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Could not read: {image_file.name}")
            continue

        patches = extract_patches(image, mask, image_file.stem)

        total_patches += patches

        print(f"{image_file.name}: " f"{patches} patches")

    print("\n==============================")
    print(f"Total patches: {total_patches}")
    print("==============================")


if __name__ == "__main__":
    main()
