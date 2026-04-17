"""
Entry point for training experiments.

Usage:
    # Train baseline UNet with default BCE+Dice loss
    python run_training.py --model unet --loss bce_dice

    # Train with different loss
    python run_training.py --model unet --loss focal
"""

import argparse
import torch
from models.unet_baseline import UNetBaseline
from training.train import train


def main():
    parser = argparse.ArgumentParser(description="Train polyp segmentation model")
    parser.add_argument("--model", default="unet", help="Model: unet")
    parser.add_argument(
        "--loss", default="bce_dice", help="Loss: bce, dice, bce_dice, focal"
    )
    args = parser.parse_args()

    if args.model == "unet":
        model = UNetBaseline()
        model_name = f"unet_baseline_{args.loss}"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train(model, model_name, loss_name=args.loss)


if __name__ == "__main__":
    main()
