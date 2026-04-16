import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient between prediction and ground truth.
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred   : Predicted mask tensor, values in [0, 1].
        target : Ground truth binary mask tensor.
        smooth : Smoothing factor to avoid division by zero.

    Returns:
        Dice score as float in [0, 1]. Higher is better.
    """
    pred = (pred > 0.5).float()  # Binarize predictions
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred   : Predicted mask tensor, values in [0, 1].
        target : Ground truth binary mask tensor.
        smooth : Smoothing factor to avoid division by zero.

    Returns:
        IoU score as float in [0, 1]. Higher is better.
    """
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def precision_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> float:
    """
    Compute Precision = TP / (TP + FP).
    Measures how many predicted polyp pixels are actually polyps.
    """
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    return ((tp + smooth) / (pred.sum() + smooth)).item()


def recall_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> float:
    """
    Compute Recall = TP / (TP + FN).
    Measures how many actual polyp pixels were correctly detected.
    """
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    return ((tp + smooth) / (target.sum() + smooth)).item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute all segmentation metrics in one call.

    Args:
        pred   : Predicted mask batch (B, 1, H, W).
        target : Ground truth mask batch (B, 1, H, W).

    Returns:
        Dictionary with dice, iou, precision, recall scores.
    """
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
    }
