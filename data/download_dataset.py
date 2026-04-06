"""
Download and prepare CVC-ClinicDB dataset.

Usage:
    python data/download_dataset.py

Requirements:
    - Kaggle API credentials set up (~/.kaggle/kaggle.json)
    - Or manually download from:
      https://www.kaggle.com/datasets/balraj98/cvcclinicdb
"""

import os
import zipfile
import argparse
from constants import *

def setup_kaggle():
    """Verify Kaggle credentials exist."""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        print("=" * 60)
        print("Kaggle credentials not found!")
        print("Steps to fix:")
        print("1. Go to kaggle.com → Account → API → Create New Token")
        print("2. Move kaggle.json to ~/.kaggle/kaggle.json")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("=" * 60)
        return False
    return True


def download_dataset(output_dir=RAW_DATA_DIR):
    """Download CVC-ClinicDB from Kaggle."""

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, ZIP_FILENAME)

    # Skip if already downloaded
    if os.path.exists(zip_path):
        print(f"Zip already exists at {zip_path}, skipping download.")
    else:
        print("Downloading CVC-ClinicDB from Kaggle...")
        result = os.system(
            f"kaggle datasets download -d {DATASET_NAME} -p {output_dir}"
        )
        if result != 0:
            print("Download failed. Check your Kaggle credentials.")
            return False
        print("Download complete.")

    return True


def unzip_dataset(output_dir=RAW_DATA_DIR):
    """Unzip the downloaded dataset."""
    zip_path = os.path.join(output_dir, ZIP_FILENAME)

    if not os.path.exists(zip_path):
        print(f"Zip file not found at {zip_path}")
        return False

    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Unzipped to {output_dir}/")

    return True


def verify_structure(output_dir=RAW_DATA_DIR):
    """Verify expected folders exist after unzip."""
    print("\nVerifying dataset structure...")

    # List what was extracted
    contents = os.listdir(output_dir)
    print(f"Contents of {output_dir}/:")
    for item in sorted(contents):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  📁 {item}/ ({count} files)")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  📄 {item} ({size_mb:.1f} MB)")

    # Count images and masks
    image_dir = os.path.join(output_dir, IMAGE_FOLDER_NAME)
    mask_dir = os.path.join(output_dir, MASK_FOLDER_NAME)

    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        n_images = len(os.listdir(image_dir))
        n_masks = len(os.listdir(mask_dir))
        print(f"\n✅ Images found: {n_images}")
        print(f"✅ Masks found:  {n_masks}")

        if n_images == n_masks == EXPECTED_IMAGES:
            print(
                f"✅ Dataset complete — {EXPECTED_IMAGES} image/mask pairs confirmed."
            )
        elif n_images != n_masks:
            print("⚠️  WARNING: Image and mask counts don't match!")
        else:
            print(
                f"⚠️  Expected {EXPECTED_IMAGES} files, got {n_images}. Check download."
            )
    else:
        print("\n⚠️  Could not find 'Original' or 'Ground Truth' folders.")
        print("   The zip may extract to a different structure.")
        print("   Check contents above and update paths in dataset.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download CVC-ClinicDB polyp segmentation dataset"
    )
    parser.add_argument(
        "--output_dir",
        default=RAW_DATA_DIR,
        help="Directory to save the dataset (default: data/raw)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, just unzip existing file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CVC-ClinicDB Dataset Download Script")
    print("=" * 60)

    if not args.skip_download:
        if not setup_kaggle():
            return
        if not download_dataset(args.output_dir):
            return

    if not unzip_dataset(args.output_dir):
        return

    verify_structure(args.output_dir)

    print("\n" + "=" * 60)
    print("Done! Next step: python data/split.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
