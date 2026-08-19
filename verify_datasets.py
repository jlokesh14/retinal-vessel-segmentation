from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "raw"


def unique_stem_files(folder, extensions):
    files = [
        f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in extensions
    ]

    unique = {}

    for f in files:
        # Treat im0001.ppm and im0001.ppm.png as the same image
        name = f.name

        if name.endswith(".ppm.png"):
            key = name[:-8]
        else:
            key = f.stem

        unique[key] = f

    return list(unique.values())


def check_dataset(name, image_dir, mask_dir, image_extensions, mask_extensions):
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    images = unique_stem_files(image_dir, image_extensions)
    masks = unique_stem_files(mask_dir, mask_extensions)

    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")

    if len(images) == len(masks):
        print("✓ Image/mask counts match")
    else:
        print("✗ Image/mask counts DO NOT match")

    if images:
        print(f"Example image: {images[0].name}")

    if masks:
        print(f"Example mask : {masks[0].name}")


check_dataset(
    "DRIVE - Training",
    ROOT / "DRIVE" / "training" / "images",
    ROOT / "DRIVE" / "training" / "1st_manual",
    {".tif", ".jpg", ".jpeg", ".png"},
    {".tif", ".gif", ".png"},
)

check_dataset(
    "DRIVE - Test",
    ROOT / "DRIVE" / "test" / "images",
    ROOT / "DRIVE" / "test" / "mask",
    {".tif", ".jpg", ".jpeg", ".png"},
    {".tif", ".gif", ".png"},
)

check_dataset(
    "STARE",
    ROOT / "STARE" / "images",
    ROOT / "STARE" / "masks",
    {".ppm", ".png"},
    {".ppm", ".png"},
)

check_dataset(
    "CHASE_DB1",
    ROOT / "CHASE_DB1" / "images",
    ROOT / "CHASE_DB1" / "masks",
    {".tif", ".tiff", ".png"},
    {".tif", ".tiff", ".png"},
)

print("\nDataset verification finished.")
