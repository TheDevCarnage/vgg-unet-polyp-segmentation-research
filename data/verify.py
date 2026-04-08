import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import PolypDataset
from data.augmentation import get_train_transforms
from constants import IMAGE_DIR_TIF, MASK_DIR_TIF, BATCH_SIZE, RESULTING_FIGURES_DIR

# ImageNet stats for denormalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for display purposes only."""
    return (tensor * STD + MEAN).clamp(0, 1)


def verify_raw():
    """
    Verify raw data loading without any transforms.
    Use this to confirm images and masks load correctly.
    """
    print("=" * 50)
    print("VERIFY MODE: No transforms")
    print("=" * 50)

    dataset = PolypDataset(
        image_dir=IMAGE_DIR_TIF, mask_dir=MASK_DIR_TIF, transform=None  # No transform
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(loader))

    print(f"✅ Dataset size : {len(dataset)} pairs")
    print(f"✅ Images shape : {images.shape}")
    print(f"✅ Masks shape  : {masks.shape}")
    print(f"✅ Mask values  : {masks.unique()}")
    print(f"✅ Image range  : [{images.min():.2f}, {images.max():.2f}]")

    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    fig.suptitle("Raw Verification — No Transforms", fontsize=14)

    for i in range(4):
        # No denormalize needed — raw pixels already [0, 1]
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs(RESULTING_FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{RESULTING_FIGURES_DIR}/verify_raw.png", dpi=150)
    print(f"\n✅ Saved → {RESULTING_FIGURES_DIR}/verify_raw.png")


def verify_with_transforms():
    """
    Verify pipeline with full training transforms applied.
    Use this to confirm augmentations and normalization work correctly.
    """
    print("=" * 50)
    print("VERIFY MODE: With transforms + denormalize")
    print("=" * 50)

    dataset = PolypDataset(
        image_dir=IMAGE_DIR_TIF, mask_dir=MASK_DIR_TIF, transform=get_train_transforms()
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    images, masks = next(iter(loader))

    print(f"✅ Dataset size : {len(dataset)} pairs")
    print(f"✅ Images shape : {images.shape}")
    print(f"✅ Masks shape  : {masks.shape}")
    print(f"✅ Mask values  : {masks.unique()}")
    # After normalization this will be negative — that is expected
    print(f"✅ Image range  : [{images.min():.2f}, {images.max():.2f}]")

    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    fig.suptitle("Transformed Verification — With Augmentations", fontsize=14)

    for i in range(4):
        # Must denormalize — transform applied ImageNet normalization
        img_display = denormalize(images[i]).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs(RESULTING_FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{RESULTING_FIGURES_DIR}/verify_transforms.png", dpi=150)
    print(f"\n✅ Saved → {RESULTING_FIGURES_DIR}/verify_transforms.png")
    print("pipeline verified!✅")


if __name__ == "__main__":
    verify_raw()  
    verify_with_transforms() 
