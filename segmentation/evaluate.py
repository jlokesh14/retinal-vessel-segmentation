from pathlib import Path
import cv2
import numpy as np

from .inference import VesselSegmenter


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "processed" / "DRIVE" / "training" / "images"
GROUND_TRUTH_DIR = ROOT / "data" / "raw" / "DRIVE" / "training" / "1st_manual"

VALIDATION_IDS = sorted([
    p.stem.replace("_training", "")
    for p in IMAGE_DIR.glob("*_training.png")
])

THRESHOLDS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def calculate_metrics(prediction, ground_truth):
    prediction = prediction > 0
    ground_truth = ground_truth > 127

    tp = np.logical_and(prediction, ground_truth).sum()
    tn = np.logical_and(~prediction, ~ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0 else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0 else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
    )

    return dice, iou, precision, recall, accuracy


def main():

    print()
    print("=" * 78)
    print("DRIVE THRESHOLD SWEEP")
    print("=" * 78)
    print("Images:", len(VALIDATION_IDS))
    print("Thresholds:", THRESHOLDS)

    segmenter = VesselSegmenter()

    # Store probability maps once so we don't run the model repeatedly.
    cached_results = []

    for image_id in VALIDATION_IDS:

        image_path = IMAGE_DIR / f"{image_id}_training.png"
        ground_truth_path = (
            GROUND_TRUTH_DIR / f"{image_id}_manual1.gif"
        )

        if not image_path.exists():
            print(f"Skipping {image_id}: image not found.")
            continue

        if not ground_truth_path.exists():
            print(f"Skipping {image_id}: ground truth not found.")
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        ground_truth = cv2.imread(
            str(ground_truth_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or ground_truth is None:
            print(f"Skipping {image_id}: could not read files.")
            continue

        _, probability = segmenter.predict(
            image,
            return_probability=True
        )

        if probability.shape != ground_truth.shape:
            probability = cv2.resize(
                probability,
                (
                    ground_truth.shape[1],
                    ground_truth.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        cached_results.append(
            (image_id, probability, ground_truth)
        )

    if not cached_results:
        raise RuntimeError(
            "No validation images were successfully evaluated."
        )

    print()
    print("-" * 78)
    print(
        f"{'Threshold':>10} "
        f"{'Dice':>10} "
        f"{'IoU':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Accuracy':>10}"
    )
    print("-" * 78)

    results = []

    for threshold in THRESHOLDS:

        all_metrics = []

        for image_id, probability, ground_truth in cached_results:

            prediction = (
                probability >= threshold
            ).astype(np.uint8) * 255

            metrics = calculate_metrics(
                prediction,
                ground_truth
            )

            all_metrics.append(metrics)

        averages = np.mean(
            np.array(all_metrics),
            axis=0
        )

        dice, iou, precision, recall, accuracy = averages

        results.append(
            (
                threshold,
                dice,
                iou,
                precision,
                recall,
                accuracy
            )
        )

        print(
            f"{threshold:10.2f} "
            f"{dice:10.4f} "
            f"{iou:10.4f} "
            f"{precision:12.4f} "
            f"{recall:10.4f} "
            f"{accuracy:10.4f}"
        )

    best = max(
        results,
        key=lambda x: x[1]
    )

    print()
    print("=" * 78)
    print("BEST THRESHOLD BY DICE")
    print("=" * 78)

    print(f"Threshold : {best[0]:.2f}")
    print(f"Dice      : {best[1]:.4f} ({best[1] * 100:.2f}%)")
    print(f"IoU       : {best[2]:.4f} ({best[2] * 100:.2f}%)")
    print(f"Precision : {best[3]:.4f} ({best[3] * 100:.2f}%)")
    print(f"Recall    : {best[4]:.4f} ({best[4] * 100:.2f}%)")
    print(f"Accuracy  : {best[5]:.4f} ({best[5] * 100:.2f}%)")

    print("=" * 78)


if __name__ == "__main__":
    main()
