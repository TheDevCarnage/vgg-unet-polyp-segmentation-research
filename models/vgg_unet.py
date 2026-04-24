import torch
import torch.nn as nn
from torchvision import models


def decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Two consecutive Conv2d -> BatchNorm -> ReLU layers.

    Serves as the fundamental building block for all decoder stages.
    Architecture is intentionally identical to the UNet baseline decoder
    to ensure that performance differences between models are attributable
    solely to encoder initialization, not decoder capacity.

    Args:
        in_channels  : Number of input channels (upsampled + skip concatenated).
        out_channels : Number of output channels after convolution.

    Returns:
        nn.Sequential block with two Conv-BN-ReLU units.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# ── VGG Feature Index Maps ─────────────────────────────────────────────────────
# VGG16 features() has 31 layers (indices 0-30):
#   Block 1 : feats[0:4]   Conv(3,64),   ReLU, Conv(64,64),   ReLU  → 64ch
#   Pool  1 : feats[4]     MaxPool2d (excluded — applied manually in forward)
#   Block 2 : feats[5:9]   Conv(64,128), ReLU, Conv(128,128), ReLU  → 128ch
#   Pool  2 : feats[9]     MaxPool2d
#   Block 3 : feats[10:16] 3x Conv(128->256) + ReLU                 → 256ch
#   Pool  3 : feats[16]    MaxPool2d
#   Block 4 : feats[17:23] 3x Conv(256->512) + ReLU                 → 512ch
#   Pool  4 : feats[23]    MaxPool2d
#   Block 5 : feats[24:30] 3x Conv(512->512) + ReLU                 → 512ch
#   Pool  5 : feats[30]    MaxPool2d
#
# VGG19 features() has 37 layers (indices 0-36):
#   Block 1 : feats[0:4]   same as VGG16                            → 64ch
#   Pool  1 : feats[4]     MaxPool2d
#   Block 2 : feats[5:9]   same as VGG16                            → 128ch
#   Pool  2 : feats[9]     MaxPool2d
#   Block 3 : feats[10:18] 4x Conv(128->256) + ReLU                 → 256ch
#   Pool  3 : feats[18]    MaxPool2d
#   Block 4 : feats[19:27] 4x Conv(256->512) + ReLU                 → 512ch
#   Pool  4 : feats[27]    MaxPool2d
#   Block 5 : feats[28:36] 4x Conv(512->512) + ReLU                 → 512ch
#   Pool  5 : feats[36]    MaxPool2d
# ──────────────────────────────────────────────────────────────────────────────

VGG_CONFIGS = {
    "vgg16": {
        "blocks": [(0, 4), (5, 9), (10, 16), (17, 23), (24, 30)],
        "weights": "IMAGENET1K_V1",
    },
    "vgg19": {
        "blocks": [(0, 4), (5, 9), (10, 18), (19, 27), (28, 36)],
        "weights": "IMAGENET1K_V1",
    },
}


class VGGUNet(nn.Module):
    """
    VGG-UNet Hybrid Architecture for Binary Polyp Segmentation.

    Replaces the randomly initialized U-Net encoder with a pretrained
    VGG16 or VGG19 backbone. The decoder mirrors the standard U-Net
    baseline architecture to ensure controlled comparison — the only
    experimental variable is the encoder initialization strategy.

    Skip connections are extracted from the output of each VGG convolutional
    block (before MaxPool) and concatenated with the corresponding decoder
    feature maps, preserving fine-grained spatial details critical for
    accurate polyp boundary delineation.

    Decoder Channel Dimensions (identical for VGG16 and VGG19):
        Stage 5 : ConvTranspose(1024->512) + skip(512) = 1024 -> 512
        Stage 4 : ConvTranspose(512->256)  + skip(512) = 768  -> 256
        Stage 3 : ConvTranspose(256->128)  + skip(256) = 384  -> 128
        Stage 2 : ConvTranspose(128->64)   + skip(128) = 192  -> 64
        Stage 1 : ConvTranspose(64->32)    + skip(64)  = 96   -> 32

    Args:
        out_channels   : Number of output segmentation channels (default: 1).
        pretrained     : Whether to load ImageNet pretrained weights (default: True).
        freeze_encoder : Whether to freeze encoder weights during training.
                         When True, only the decoder and bottleneck are updated.
                         Useful for ablation studies on fine-tuning strategy.
        backbone       : VGG variant to use as encoder. One of 'vgg16' or 'vgg19'.
                         Both backbones produce identical output channel dimensions,
                         enabling direct architectural comparison (default: 'vgg16').

    Input:  (B, 3, H, W) — must be normalized with ImageNet mean/std
    Output: (B, 1, H, W) — sigmoid-activated binary segmentation mask

    Note:
        Input normalization must use ImageNet statistics:
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        This is handled automatically by get_val_transforms() in augmentation.py.
    """

    def __init__(
        self,
        out_channels: int = 1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        backbone: str = "vgg16",
    ):
        super(VGGUNet, self).__init__()

        if backbone not in VGG_CONFIGS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(VGG_CONFIGS.keys())}"
            )

        # ── Encoder: Pretrained VGG Backbone ──────────────────
        config = VGG_CONFIGS[backbone]
        weights = config["weights"] if pretrained else None

        if backbone == "vgg16":
            vgg = models.vgg16(weights=weights)
        else:
            vgg = models.vgg19(weights=weights)

        feats = list(vgg.features.children())
        b1, b2, b3, b4, b5 = config["blocks"]

        # Extract convolutional blocks — MaxPool layers are excluded here
        # and applied manually in forward() to control skip connection timing
        self.enc1 = nn.Sequential(*feats[b1[0] : b1[1]])  # -> 64ch
        self.enc2 = nn.Sequential(*feats[b2[0] : b2[1]])  # -> 128ch
        self.enc3 = nn.Sequential(*feats[b3[0] : b3[1]])  # -> 256ch
        self.enc4 = nn.Sequential(*feats[b4[0] : b4[1]])  # -> 512ch
        self.enc5 = nn.Sequential(*feats[b5[0] : b5[1]])  # -> 512ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Freeze encoder weights if performing frozen-encoder ablation
        if freeze_encoder:
            encoder_modules = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
            for module in encoder_modules:
                for param in module.parameters():
                    param.requires_grad = False

        # ── Bottleneck ────────────────────────────────────────
        # Applied at the lowest resolution (H/32 x W/32)
        # Expands channels from 512 to 1024 before decoding begins
        self.bottleneck = decoder_block(512, 1024)

        # ── Decoder ───────────────────────────────────────────
        # Each stage: ConvTranspose2d (upsample) -> concat skip -> decoder_block
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = decoder_block(512 + 512, 512)  # cat(up5, enc5) -> 512

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = decoder_block(256 + 512, 256)  # cat(up4, enc4) -> 256

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = decoder_block(128 + 256, 128)  # cat(up3, enc3) -> 128

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = decoder_block(64 + 128, 64)  # cat(up2, enc2) -> 64

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = decoder_block(32 + 64, 32)  # cat(up1, enc1) -> 32

        # ── Output Head ───────────────────────────────────────
        # 1x1 convolution projects to desired number of output channels
        # Sigmoid produces probability map in [0, 1]
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VGG encoder and UNet-style decoder.

        Encoder extracts hierarchical feature maps at five resolution levels.
        Skip connections bridge each encoder level to the corresponding decoder
        stage, preserving spatial details lost during downsampling.

        Args:
            x: ImageNet-normalized input tensor of shape (B, 3, H, W).

        Returns:
            Sigmoid-activated binary segmentation mask of shape (B, 1, H, W).
        """
        # ── Encoder ───────────────────────────────────────────
        e1 = self.enc1(x)  # (B, 64,  H,    W   )
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2,  W/2 )
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4,  W/4 )
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8,  W/8 )
        e5 = self.enc5(self.pool(e4))  # (B, 512, H/16, W/16)

        # ── Bottleneck ────────────────────────────────────────
        b = self.bottleneck(self.pool(e5))  # (B, 1024, H/32, W/32)

        # ── Decoder with Skip Connections ─────────────────────
        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))  # (B, 512, H/16, W/16)
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))  # (B, 256, H/8,  W/8 )
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, H/4,  W/4 )
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64,  H/2,  W/2 )
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32,  H,    W   )

        # ── Output ────────────────────────────────────────────
        return torch.sigmoid(self.output(d1))  # (B, 1, H, W)


if __name__ == "__main__":
    for backbone in ["vgg16", "vgg19"]:
        print(f"\n{'='*50}")
        print(f"  Backbone: {backbone}")
        print(f"{'='*50}")

        model = VGGUNet(pretrained=True, backbone=backbone)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        print(f"  Input  shape : {x.shape}")
        print(f"  Output shape : {output.shape}")
        print(f"  Output range : [{output.min():.4f}, {output.max():.4f}]")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  Total params : {total:,}")
        print(f"  Trainable    : {trainable:,}")
        print(f"  Frozen       : {frozen:,}")
