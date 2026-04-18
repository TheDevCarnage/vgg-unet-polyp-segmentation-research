import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PolypDataset
from data.augmentation import get_train_transforms, get_val_transforms
from training.loss import get_loss
from training.metrics import compute_all_metrics
from constants import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    VAL_IMG_DIR,   VAL_MASK_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    CHECKPOINT_DIR, RESULTS_DIR,
)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics"   : metrics,
    }, path)
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint — returns epoch and metrics."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"  ✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], checkpoint["metrics"]


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full training epoch.

    Returns:
        avg_loss : Mean loss over all batches.
        metrics  : Mean dice and iou over all batches.
    """
    model.train()
    total_loss = 0.0
    all_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    loop = tqdm(loader, desc="  Train", leave=False)

    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        batch_metrics = compute_all_metrics(preds.detach(), masks)
        for k in all_metrics:
            all_metrics[k] += batch_metrics[k]

        loop.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Run validation — no gradient computation.

    Returns:
        avg_loss : Mean validation loss.
        metrics  : Mean dice and iou over all batches.
    """
    model.eval()
    total_loss  = 0.0
    all_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    loop = tqdm(loader, desc="  Val  ", leave=False)

    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        preds = model(images)
        loss  = criterion(preds, masks)

        total_loss += loss.item()
        batch_metrics = compute_all_metrics(preds, masks)
        for k in all_metrics:
            all_metrics[k] += batch_metrics[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


def train(model, model_name: str, loss_name: str = "bce_dice"):
    """
    Full training loop with checkpointing and logging.

    Args:
        model      : PyTorch model to train.
        model_name : Name for saving checkpoints and logs (e.g. 'unet_baseline').
        loss_name  : Loss function to use — see training/loss.py registry.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} for training")
    print(f"\n{'='*55}")
    print(f"  Model    : {model_name}")
    print(f"  Loss     : {loss_name}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {NUM_EPOCHS}")
    print(f"{'='*55}\n")

    model = model.to(device)

    # Data loaders
    train_dataset = PolypDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, get_train_transforms())
    val_dataset   = PolypDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   get_val_transforms())

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}\n")

    # Baseline UNet — uniform LR
    # Optimizer + scheduler + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    is_vgg = hasattr(model, "enc1") and hasattr(model, "bottleneck")

    if is_vgg:
        # VGG encoder params
        encoder_params = (
            list(model.enc1.parameters())
            + list(model.enc2.parameters())
            + list(model.enc3.parameters())
            + list(model.enc4.parameters())
            + list(model.enc5.parameters())
        )
        # Decoder + bottleneck params
        decoder_params = (
            list(model.bottleneck.parameters())
            + list(model.up5.parameters())
            + list(model.dec5.parameters())
            + list(model.up4.parameters())
            + list(model.dec4.parameters())
            + list(model.up3.parameters())
            + list(model.dec3.parameters())
            + list(model.up2.parameters())
            + list(model.dec2.parameters())
            + list(model.up1.parameters())
            + list(model.dec1.parameters())
            + list(model.output.parameters())
        )
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": LEARNING_RATE / 10},  # 1e-5
                {"params": decoder_params, "lr": LEARNING_RATE},  # 1e-4
            ]
        )
        print(
            f"  Differential LR → encoder: {LEARNING_RATE/10}, decoder: {LEARNING_RATE}"
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )
    criterion = get_loss(loss_name)

    # Tracking
    best_dice    = 0.0
    history      = []
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start

        # Step scheduler on validation dice
        scheduler.step(val_metrics["dice"])

        # Log results
        row = {
            "epoch"         : epoch,
            "train_loss"    : round(train_loss, 4),
            "val_loss"      : round(val_loss, 4),
            "train_dice"    : round(train_metrics["dice"], 4),
            "val_dice"      : round(val_metrics["dice"], 4),
            "train_iou"     : round(train_metrics["iou"], 4),
            "val_iou"       : round(val_metrics["iou"], 4),
            "val_precision" : round(val_metrics["precision"], 4),
            "val_recall"    : round(val_metrics["recall"], 4),
        }
        history.append(row)

        print(
            f"  Loss  → train: {train_loss:.4f}  val: {val_loss:.4f}\n"
            f"  Dice  → train: {train_metrics['dice']:.4f}  val: {val_metrics['dice']:.4f}\n"
            f"  IoU   → train: {train_metrics['iou']:.4f}   val: {val_metrics['iou']:.4f}\n"
            f"  Time  → {elapsed:.1f}s\n"
        )

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            print(f"New best Dice: {best_dice:.4f}\n")

    # Save training history to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\n📊 Training history saved → {history_path}")
    print(f"Best Val Dice : {best_dice:.4f}")
    print(f"{'='*55}")

    return history
