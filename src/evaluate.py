from pathlib import Path
import cv2
import numpy as np
import torch
from models.unet import UNet

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "processed" / "DRIVE" / "training" / "images"
VESSEL_MASK_DIR = ROOT / "data" / "raw" / "DRIVE" / "training" / "1st_manual"
FOV_MASK_DIR = ROOT / "data" / "raw" / "DRIVE" / "training" / "mask"
CHECKPOINT = ROOT / "checkpoints" / "best_unet.pth"

PATCH_SIZE = 64
STRIDE = 64


def dice_score(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    return (2.0 * intersection + 1.0) / (prediction.sum() + target.sum() + 1.0)


def iou_score(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return (intersection + 1.0) / (union + 1.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    dice_scores = []
    iou_scores = []

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    print("=" * 60)
    print("DRIVE TRAINING-SET VESSEL EVALUATION")
    print("=" * 60)
    print("Device:", device)
    print("Images:", len(image_files))
    print("Best checkpoint epoch:", checkpoint["epoch"])

    with torch.no_grad():

        for image_file in image_files:

            stem = image_file.stem

            vessel_mask_file = (
                VESSEL_MASK_DIR / stem.replace("_training", "_manual1")
            ).with_suffix(".gif")

            fov_mask_file = FOV_MASK_DIR / f"{stem}_mask.gif"

            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)

            vessel_mask = cv2.imread(str(vessel_mask_file), cv2.IMREAD_GRAYSCALE)

            fov_mask = cv2.imread(str(fov_mask_file), cv2.IMREAD_GRAYSCALE)

            if image is None or vessel_mask is None or fov_mask is None:
                print("Could not read:", image_file.name)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

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

                    patch = patch[:, :, 0]

                    patch = patch.astype(np.float32) / 255.0

                    tensor = torch.from_numpy(patch[None, None, :, :]).to(device)

                    logits = model(tensor)

                    probabilities = torch.sigmoid(logits)

                    prediction = probabilities[0, 0].cpu().numpy()

                    prediction_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += prediction

                    prediction_count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

            valid = prediction_count > 0

            reconstructed = np.zeros((height, width), dtype=np.float32)

            reconstructed[valid] = prediction_sum[valid] / prediction_count[valid]

            prediction = reconstructed >= 0.5

            target = vessel_mask > 127

            fov = fov_mask > 127

            prediction = prediction & fov
            target = target & fov

            dice = dice_score(prediction, target)

            iou = iou_score(prediction, target)

            dice_scores.append(dice)
            iou_scores.append(iou)

            print(f"{image_file.name}: " f"Dice={dice:.4f} | " f"IoU={iou:.4f}")

    print()
    print("=" * 60)
    print("FINAL VESSEL SEGMENTATION RESULTS")
    print("=" * 60)

    if dice_scores:

        print("Mean Dice:", round(float(np.mean(dice_scores)), 4))

        print("Mean IoU:", round(float(np.mean(iou_scores)), 4))

        print("Best Dice:", round(float(np.max(dice_scores)), 4))

        print("Worst Dice:", round(float(np.min(dice_scores)), 4))

    print("=" * 60)


if __name__ == "__main__":
    main()
