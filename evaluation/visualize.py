import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import PolypDataset
from data.augmentation import get_val_transforms
from training.metrics import compute_all_metrics
from constants import TEST_IMG_DIR, TEST_MASK_DIR, RESULTS_DIR

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    return (tensor * STD + MEAN).clamp(0, 1)


def plot_predictions(model, model_name: str, transform: str, n_samples: int = 6):
    """
    Visualize model predictions on test set.
    Saves a grid of: Input Image | Ground Truth | Prediction

    Args:
        model      : Trained PyTorch model.
        model_name : Used for saving the output file.
        n_samples  : Number of test samples to visualize.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = model.to(device).eval()

    dataset = PolypDataset(TEST_IMG_DIR, TEST_MASK_DIR, get_val_transforms())
    loader  = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    images, masks = next(iter(loader))

    with torch.no_grad():
        preds = model(images.to(device)).cpu()

    metrics = compute_all_metrics(preds, masks)
    print(f"\nSample Metrics (visualization only) — {model_name}{transform}")
    print(f"  Dice      : {metrics['dice']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    fig.suptitle(
        f"{model_name} — Dice: {metrics['dice']:.4f}  IoU: {metrics['iou']:.4f}",
        fontsize=14
    )

    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(n_samples):
        img_display  = denormalize(images[i]).permute(1, 2, 0).numpy()
        mask_display = masks[i].squeeze().numpy()
        pred_display = (preds[i].squeeze() > 0.5).float().numpy()

        axes[i, 0].imshow(img_display)
        axes[i, 1].imshow(mask_display, cmap="gray")
        axes[i, 2].imshow(pred_display, cmap="gray")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"figures/{model_name}{transform}_predictions.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Predictions saved → {save_path}")


def plot_training_history(model_name: str, transform: str):
    """
    Plot training curves from saved CSV history.
    Shows loss and dice score over epochs for train and validation.

    Args:
        model_name : Must match the name used during training.
    """
    import pandas as pd

    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.csv")
    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training History — {model_name}{transform}", fontsize=14)

    # Loss curve
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Dice curve
    axes[1].plot(df["epoch"], df["train_dice"], label="Train Dice")
    axes[1].plot(df["epoch"], df["val_dice"],   label="Val Dice")
    axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    # IoU curve
    axes[2].plot(df["epoch"], df["train_iou"], label="Train IoU")
    axes[2].plot(df["epoch"], df["val_iou"],   label="Val IoU")
    axes[2].set_title("IoU Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"figures/{model_name}{transform}_history.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Training curves saved → {save_path}")
