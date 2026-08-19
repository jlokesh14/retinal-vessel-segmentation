from pathlib import Path
import cv2
import numpy as np
import torch
from models.unet import UNet

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "processed" / "DRIVE" / "test" / "images"
FOV_DIR = ROOT / "data" / "raw" / "DRIVE" / "test" / "mask"
CHECKPOINT = ROOT / "checkpoints" / "best_unet.pth"
OUTPUT_DIR = ROOT / "outputs" / "DRIVE" / "test_predictions"

PATCH_SIZE = 64
STRIDE = 64
THRESHOLD = 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    print("=" * 60)
    print("DRIVE TEST-SET VESSEL PREDICTION")
    print("=" * 60)
    print("Device:", device)
    print("Images:", len(image_files))
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Output:", OUTPUT_DIR)

    with torch.no_grad():

        for image_file in image_files:

            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

            fov_file = FOV_DIR / f"{image_file.stem}_mask.gif"

            fov = cv2.imread(str(fov_file), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print("Could not read:", image_file.name)
                continue

            if fov is None:
                print("Could not read FOV:", fov_file.name)
                continue

            height, width = image.shape

            prediction_sum = np.zeros((height, width), dtype=np.float32)

            prediction_count = np.zeros((height, width), dtype=np.float32)

            y_positions = list(range(0, height - PATCH_SIZE + 1, STRIDE))

            x_positions = list(range(0, width - PATCH_SIZE + 1, STRIDE))

            if y_positions[-1] != height - PATCH_SIZE:
                y_positions.append(height - PATCH_SIZE)

            if x_positions[-1] != width - PATCH_SIZE:
                x_positions.append(width - PATCH_SIZE)

            for y in y_positions:
                for x in x_positions:

                    patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                    patch = patch.astype(np.float32) / 255.0

                    tensor = torch.from_numpy(patch[None, None, :, :]).to(device)

                    logits = model(tensor)

                    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

                    prediction_sum[
                        y : y + PATCH_SIZE, x : x + PATCH_SIZE
                    ] += probability

                    prediction_count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

            valid = prediction_count > 0

            reconstructed = np.zeros((height, width), dtype=np.float32)

            reconstructed[valid] = prediction_sum[valid] / prediction_count[valid]

            prediction = (reconstructed >= THRESHOLD).astype(np.uint8) * 255

            fov_binary = fov > 127

            prediction[~fov_binary] = 0

            output_file = OUTPUT_DIR / f"{image_file.stem}_prediction.png"

            cv2.imwrite(str(output_file), prediction)

            vessel_percent = (prediction > 0).mean() * 100

            print(
                f"{image_file.name}: "
                f"Vessel={vessel_percent:.2f}% -> "
                f"{output_file.name}"
            )

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
