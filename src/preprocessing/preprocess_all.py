from pathlib import Path
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def preprocess_drive_image(image_path, fov_path, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    fov = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not read image: {image_path.name}")
        return

    if fov is None:
        print(f"Could not read FOV: {fov_path.name}")
        return

    # Green channel
    green = image[:, :, 1]

    # Convert FOV to binary mask
    fov = (fov > 0).astype(np.uint8) * 255

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced = clahe.apply(green)

    # Normalize ONLY inside FOV
    pixels = enhanced[fov > 0]

    if len(pixels) > 0:
        min_value = pixels.min()
        max_value = pixels.max()

        if max_value > min_value:
            normalized = (
                (enhanced.astype(np.float32) - min_value)
                / (max_value - min_value)
                * 255
            )
        else:
            normalized = enhanced.astype(np.float32)
    else:
        normalized = enhanced.astype(np.float32)

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Remove outside-FOV pixels
    normalized[fov == 0] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), normalized)


def process_drive():
    print("\n========== DRIVE ==========")

    # Training
    image_dir = RAW / "DRIVE" / "training" / "images"
    fov_dir = RAW / "DRIVE" / "training" / "mask"
    output_dir = PROCESSED / "DRIVE" / "training" / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.glob("*.tif")):

        fov_path = fov_dir / image_path.name.replace(
            "_training.tif", "_training_mask.gif"
        ).replace("_test.tif", "_test_mask.gif")

        output_path = output_dir / f"{image_path.stem}.png"

        preprocess_drive_image(image_path, fov_path, output_path)

        print(f"Processed: {image_path.name}")

    # Test
    image_dir = RAW / "DRIVE" / "test" / "images"
    fov_dir = RAW / "DRIVE" / "test" / "mask"
    output_dir = PROCESSED / "DRIVE" / "test" / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.glob("*.tif")):

        fov_path = fov_dir / image_path.name.replace(
            "_training.tif", "_training_mask.gif"
        ).replace("_test.tif", "_test_mask.gif")

        output_path = output_dir / f"{image_path.stem}.png"

        preprocess_drive_image(image_path, fov_path, output_path)

        print(f"Processed: {image_path.name}")


def preprocess_standard_image(image_path, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not read: {image_path.name}")
        return

    # Green channel
    green = image[:, :, 1]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced = clahe.apply(green)

    # Normalize
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), normalized)


def process_stare():
    print("\n========== STARE ==========")

    input_dir = RAW / "STARE" / "images"
    output_dir = PROCESSED / "STARE" / "images"

    for image_path in sorted(input_dir.glob("*.ppm")):

        output_path = output_dir / f"{image_path.stem}.png"

        preprocess_standard_image(image_path, output_path)

        print(f"Processed: {image_path.name}")


def process_chase():
    print("\n========== CHASE_DB1 ==========")

    input_dir = RAW / "CHASE_DB1" / "images"
    output_dir = PROCESSED / "CHASE_DB1" / "images"

    for image_path in sorted(input_dir.glob("*.tif")):

        output_path = output_dir / f"{image_path.stem}.png"

        preprocess_standard_image(image_path, output_path)

        print(f"Processed: {image_path.name}")


def main():

    print("Starting corrected preprocessing...")

    process_drive()
    process_stare()
    process_chase()

    print("\n================================")
    print("Corrected preprocessing complete.")
    print("================================")


if __name__ == "__main__":
    main()
