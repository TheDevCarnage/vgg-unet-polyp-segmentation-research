import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PolypDataset
from data.augmentation import get_val_transforms
from training.metrics import compute_all_metrics
from evaluation.visualize import plot_predictions, plot_training_history
from models.unet_baseline import UNetBaseline
from models.vgg_unet import VGGUNet
from constants import TEST_IMG_DIR, TEST_MASK_DIR, RESULTS_DIR, CHECKPOINT_DIR


def evaluate(model, model_name: str, checkpoint_path: str):
    """
    Run full evaluation on test set using best saved checkpoint.
    Computes Dice, IoU, Precision, Recall and saves prediction visuals.

    Args:
        model           : PyTorch model instance.
        model_name      : Used for saving results.
        checkpoint_path : Path to saved .pth checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    print(f"\n✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Val Dice at save time: {checkpoint['metrics']['dice']:.4f}")

    # Test dataset
    test_dataset = PolypDataset(TEST_IMG_DIR, TEST_MASK_DIR, get_val_transforms())
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Accumulate metrics across all test batches
    all_metrics  = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            preds  = model(images).cpu()
            batch_metrics = compute_all_metrics(preds, masks)
            for k in all_metrics:
                all_metrics[k] += batch_metrics[k]

    # Average over all batches
    n = len(test_loader)
    final_metrics = {k: round(v / n, 4) for k, v in all_metrics.items()}

    # Print results
    print(f"\n{'='*45}")
    print(f"  TEST RESULTS — {model_name}")
    print(f"{'='*45}")
    print(f"  Dice      : {final_metrics['dice']}")
    print(f"  IoU       : {final_metrics['iou']}")
    print(f"  Precision : {final_metrics['precision']}")
    print(f"  Recall    : {final_metrics['recall']}")
    print(f"{'='*45}\n")

    # Save metrics to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_test_results.csv")
    pd.DataFrame([final_metrics]).to_csv(results_path, index=False)
    print(f"✅ Test results saved → {results_path}")

    # Plot predictions and training curves
    plot_predictions(model, model_name, n_samples=6)
    plot_training_history(model_name)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation model")
    parser.add_argument("--model",      default="unet",     help="Model: unet | vgg_unet")
    parser.add_argument("--loss",       default="bce_dice", help="Loss used during training")
    parser.add_argument("--checkpoint", default=None,       help="Path to .pth checkpoint")
    args = parser.parse_args()

    # Auto-resolve checkpoint path if not provided
    model_name      = f"{args.model}_baseline_{args.loss}"
    checkpoint_path = args.checkpoint or f"{CHECKPOINT_DIR}/{model_name}_best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    if args.model == "unet":
        model = UNetBaseline()
    elif args.model == "vgg_unet":
        model = VGGUNet()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    evaluate(model, model_name, checkpoint_path)


if __name__ == "__main__":
    main()
