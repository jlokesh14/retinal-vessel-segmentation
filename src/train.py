from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import RetinalPatchDataset
from src.models.unet import UNet


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "images"
MASK_DIR = ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "masks"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_IMAGES = 2
SEED = 42


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(probabilities.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probabilities * targets).sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)

    dice = dice_loss(logits, targets)

    return bce + dice


def dice_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        predictions.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images[:, :1, :, :].to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits.detach(), masks)
        total_iou += iou_score(logits.detach(), masks)

    n = len(loader)

    return (total_loss / n, total_dice / n, total_iou / n)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images[:, :1, :, :].to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = combined_loss(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        total_iou += iou_score(logits, masks)

    n = len(loader)

    return (total_loss / n, total_dice / n, total_iou / n)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("RETINAL VESSEL SEGMENTATION TRAINING")
    print("=" * 60)
    print("Device:", device)

    dataset = RetinalPatchDataset(IMAGE_DIR, MASK_DIR)

    # Group patches by their original DRIVE image.
    image_groups = {}

    for index, image_file in enumerate(dataset.image_files):
        source_id = image_file.stem.rsplit("_", 1)[0]
        image_groups.setdefault(source_id, []).append(index)

    source_ids = sorted(image_groups.keys())

    rng = random.Random(SEED)
    rng.shuffle(source_ids)

    validation_ids = source_ids[:VALIDATION_IMAGES]
    training_ids = source_ids[VALIDATION_IMAGES:]

    train_indices = [
        index for source_id in training_ids for index in image_groups[source_id]
    ]

    validation_indices = [
        index for source_id in validation_ids for index in image_groups[source_id]
    ]

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    print("Training images   :", len(training_ids))
    print("Validation images :", len(validation_ids))
    print("Training IDs      :", ", ".join(training_ids))
    print("Validation IDs    :", ", ".join(validation_ids))
    print("Training patches  :", len(train_dataset))
    print("Validation patches:", len(validation_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, val_dice, val_iou = validate(model, validation_loader, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            checkpoint_path = CHECKPOINT_DIR / "best_unet.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "training_ids": training_ids,
                    "validation_ids": validation_ids,
                },
                checkpoint_path,
            )

            print(f"  Saved best model -> {checkpoint_path}")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("Best validation Dice:", round(best_dice, 4))
    print("=" * 60)


if __name__ == "__main__":
    main()
