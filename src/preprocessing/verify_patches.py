from pathlib import Path
import cv2
import random

ROOT = Path(__file__).resolve().parents[2]

PATCH_ROOT = ROOT / "data" / "processed" / "DRIVE" / "training" / "patches"
IMAGE_DIR = PATCH_ROOT / "images"
MASK_DIR = PATCH_ROOT / "masks"


def main():
    image_files = sorted(IMAGE_DIR.glob("*.png"))
    mask_files = sorted(MASK_DIR.glob("*.png"))

    print("=" * 50)
    print("PATCH VERIFICATION")
    print("=" * 50)

    print(f"Image patches : {len(image_files)}")
    print(f"Mask patches  : {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print("ERROR: Image/mask counts do not match!")
        return

    print("✓ Image/mask counts match")

    # Check dimensions and readable files
    bad_images = []
    bad_masks = []

    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)

        if img is None:
            bad_images.append(img_file.name)
        elif img.shape[:2] != (64, 64):
            bad_images.append(f"{img_file.name} -> {img.shape}")

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            bad_masks.append(mask_file.name)
        elif mask.shape != (64, 64):
            bad_masks.append(f"{mask_file.name} -> {mask.shape}")

    print(f"Bad image patches: {len(bad_images)}")
    print(f"Bad mask patches : {len(bad_masks)}")

    if bad_images:
        print("Image problems:")
        for x in bad_images[:10]:
            print(" ", x)

    if bad_masks:
        print("Mask problems:")
        for x in bad_masks[:10]:
            print(" ", x)

    # Show a random matching pair
    if image_files:
        img_file = random.choice(image_files)

        # Patch names should correspond
        mask_file = MASK_DIR / img_file.name

        img = cv2.imread(str(img_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        print()
        print("Random patch:")
        print("Image:", img_file.name)
        print("Image shape:", img.shape)
        print("Mask shape :", mask.shape)
        print("Mask values:", sorted(set(mask.flatten().tolist()))[:20])

    print()
    print("=" * 50)
    print("Verification finished.")
    print("=" * 50)


if __name__ == "__main__":
    main()
