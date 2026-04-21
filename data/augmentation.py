import albumentations as A


def get_train_transforms(height=256, width=256):
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.GaussianBlur(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],  # ImageNet std
            ),
        ]
    )


def get_val_transforms(height=256, width=256):
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_train_transforms_minimal(height=256, width=256):
    """Minimal augmentation — only resize and normalize."""
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_train_transforms_heavy(height=256, width=256):
    """
    Heavy augmentation — additional elastic and grid distortions.
    Simulates real colonoscopy variation in polyp shape and texture.
    """
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
            A.GaussianBlur(p=0.3),
            A.ElasticTransform(p=0.3),  # Polyp boundary deformation simulation
            A.GridDistortion(p=0.2),  # Colonoscope lens distortion simulation
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
