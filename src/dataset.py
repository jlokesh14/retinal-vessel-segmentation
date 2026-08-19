from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RetinalPatchDataset(Dataset):
    """
    PyTorch dataset for retinal vessel segmentation patches.

    Images:
        64x64 RGB PNG

    Masks:
        64x64 grayscale PNG
        0   = background
        255 = vessel
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_files = sorted(self.image_dir.glob("*.png"))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image patches found in {self.image_dir}")

        # Make sure every image has a corresponding mask
        for image_file in self.image_files:
            mask_file = self.mask_dir / image_file.name

            if not mask_file.exists():
                raise RuntimeError(f"Missing mask for {image_file.name}")

        print(f"Dataset: {self.image_dir}")
        print(f"Images : {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = self.mask_dir / image_file.name

        # Read image
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_file}")

        # OpenCV loads BGR -> convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask as grayscale
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_file}")

        # Convert image from:
        # H x W x C
        # to:
        # C x H x W
        image = image.transpose(2, 0, 1)

        # Normalize image from [0,255] -> [0,1]
        image = image.astype(np.float32) / 255.0

        # Convert mask from [0,255] -> [0,1]
        mask = (mask > 127).astype(np.float32)

        # Add channel dimension:
        # H x W -> 1 x H x W
        mask = np.expand_dims(mask, axis=0)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]

    image_dir = (
        ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "images"
    )

    mask_dir = ROOT / "data" / "processed" / "DRIVE" / "training" / "patches" / "masks"

    dataset = RetinalPatchDataset(image_dir, mask_dir)

    print()
    print("=" * 50)
    print("DATASET TEST")
    print("=" * 50)

    print("Dataset length:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min:", image.min().item())
    print("Image max:", image.max().item())

    print()

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask min:", mask.min().item())
    print("Mask max:", mask.max().item())

    print()
    print("Dataset test completed.")
