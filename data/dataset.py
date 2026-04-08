import os
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset
from constants import SUPPORTED_FORMATS

class PolypDataset(Dataset):
    """
    PyTorch Dataset for the CVC-ClinicDB polyp segmentation dataset.

    Loads paired colonoscopy images and binary segmentation masks from
    TIF format. Handles mixed image orientations and optional augmentations
    via an Albumentations transform pipeline.

    Args:
        image_dir (str): Path to directory containing colonoscopy images.
        mask_dir  (str): Path to directory containing binary mask files.
        transform (callable, optional): Albumentations transform pipeline.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect only valid image files, sorted for deterministic pairing
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(SUPPORTED_FORMATS)]
        )
        self.masks = sorted(
            [f for f in os.listdir(mask_dir) if f.endswith(SUPPORTED_FORMATS)]
        )

        assert len(self.images) == len(
            self.masks
        ), f"Mismatch: {len(self.images)} images but {len(self.masks)} masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load TIF files using tifffile
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        # Normalize portrait orientation to landscape
        # Some CVC-ClinicDB TIFs are stored rotated
        if image.shape[0] > image.shape[1]:
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()

        # Ensure image is 3-channel RGB (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]  # Drop alpha channel if present

        # Ensure mask is single-channel grayscale (H, W)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Cast to uint8 — required by Albumentations
        image = image.astype(np.uint8)

        # Binarize mask: polyp pixels = 1.0, background = 0.0
        mask = (mask > 0).astype(np.float32)

        # Apply augmentation pipeline if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Returns uint8 numpy array
            mask = augmented["mask"]  # Returns float32 numpy array

        # Convert to tensors
        # NOTE: Division by 255.0 only if transform has NO Normalize step
        # If using A.Normalize — it handles scaling internally, do NOT divide
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
