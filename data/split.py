import os
import shutil
import random

from constants import IMAGE_DIR_TIF, MASK_DIR_TIF, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO




def split_dataset(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    train: float = TRAIN_RATIO,
    val: float = VAL_RATIO,
    test: float = TEST_RATIO,
    seed: int = 42,
) -> None:
    """
    Split the CVC-ClinicDB dataset into train, validation, and test sets.

    Copies image-mask pairs into the following structure:
        output_dir/
            train/images/  train/masks/
            val/images/    val/masks/
            test/images/   test/masks/

    Args:
        image_dir  : Source directory containing colonoscopy images.
        mask_dir   : Source directory containing binary mask files.
        output_dir : Root directory for the split output.
        train      : Fraction of data for training (default: 0.8).
        val        : Fraction of data for validation (default: 0.1).
        test       : Fraction of data for testing (default: 0.1).
        seed       : Random seed for reproducibility (default: 42).
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Seed for reproducibility — same split every run
    random.seed(seed)
    images = sorted(os.listdir(image_dir))
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train)
    val_end = int(n * (train + val))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    print("=" * 40)
    print("Splitting dataset...")
    print("=" * 40)

    for split, files in splits.items():
        img_out = os.path.join(output_dir, split, "images")
        mask_out = os.path.join(output_dir, split, "masks")

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for f in files:
            shutil.copy(os.path.join(image_dir, f), os.path.join(img_out, f))
            shutil.copy(os.path.join(mask_dir, f), os.path.join(mask_out, f))

        print(f"  {split:5s} : {len(files)} image-mask pairs")

    print("=" * 40)
    print(f"Split complete → {output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    split_dataset(
        image_dir=IMAGE_DIR_TIF,
        mask_dir=MASK_DIR_TIF,
        output_dir=OUTPUT_DIR,
    )
