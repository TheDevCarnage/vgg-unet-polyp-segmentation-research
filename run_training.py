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
from models.vgg_unet import VGGUNet
from training.train import train


def main():
    parser = argparse.ArgumentParser(description="Train polyp segmentation model")
    parser.add_argument("--model", default="unet", help="Model: unet | vgg_unet | vgg_unet_frozen")
    parser.add_argument(
        "--loss", default="bce_dice", help="Loss: bce, dice, bce_dice, focal"
    )
    args = parser.parse_args()

    if args.model == "unet":
        model = UNetBaseline()
        model_name = f"unet_baseline_{args.loss}"
    elif args.model == "vgg_unet":
        model = VGGUNet(pretrained=True)
        model_name = f"vgg_unet_{args.loss}"
    elif args.model == "vgg_unet_frozen":
        model = VGGUNet(pretrained=True, freeze_encoder=True)
        model_name = f"vgg_unet_frozen_{args.loss}"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train(model, model_name, loss_name=args.loss)


if __name__ == "__main__":
    main()
