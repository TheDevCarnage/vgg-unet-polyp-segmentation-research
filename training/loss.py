import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Measures overlap between prediction and ground truth.
    Best for imbalanced datasets where polyp pixels << background pixels.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.

    BCE handles pixel-wise accuracy.
    Dice handles region overlap.
    Together they give more stable training for medical segmentation.

    Args:
        bce_weight  : Weight for BCE component (default: 0.5).
        dice_weight : Weight for Dice component (default: 0.5).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard ones.
    Useful when polyps are very small and easy negatives dominate.

    Args:
        alpha : Weighting factor for positive class (default: 0.8).
        gamma : Focusing parameter (default: 2.0).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# Loss registry — add new losses here for experiments
LOSS_REGISTRY = {
    "bce": nn.BCELoss,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "focal": FocalLoss,
}


def get_loss(name: str) -> nn.Module:
    """
    Retrieve loss function by name from registry.

    Args:
        name: One of 'bce', 'dice', 'bce_dice', 'focal'.

    Returns:
        Instantiated loss function.
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]()
