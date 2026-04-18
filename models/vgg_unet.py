import torch
import torch.nn as nn
from torchvision import models


def decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Two consecutive Conv2d → BatchNorm → ReLU layers.
    Fundamental decoder building block — mirrors UNet baseline.

    Args:
        in_channels  : Input channels (upsampled + skip concatenated).
        out_channels : Output channels after convolution.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class VGGUNet(nn.Module):
    """
    VGG16-UNet Hybrid Architecture for Binary Polyp Segmentation.

    Uses pretrained VGG16 as encoder and a custom UNet-style decoder.
    Skip connections from each VGG block feed into the decoder for
    spatial detail preservation — critical for polyp boundary accuracy.

    VGG16 Feature Map Indices (31 layers total):
        Block 1 : feats[0:4]   → 2x Conv (64ch)  + ReLU
        Pool 1  : feats[4]     → MaxPool (skipped — handled manually)
        Block 2 : feats[5:9]   → 2x Conv (128ch) + ReLU
        Pool 2  : feats[9]     → MaxPool (skipped)
        Block 3 : feats[10:17] → 3x Conv (256ch) + ReLU
        Pool 3  : feats[17]    → MaxPool (skipped)
        Block 4 : feats[18:25] → 3x Conv (512ch) + ReLU
        Pool 4  : feats[25]    → MaxPool (skipped)
        Block 5 : feats[26:31] → 3x Conv (512ch) + ReLU

    Decoder Channel Math:
        dec5 : up(1024) + skip(512) = 1024 → 512
        dec4 : up(512)  + skip(512) = 1024 → 256
        dec3 : up(256)  + skip(256) = 512  → 128
        dec2 : up(128)  + skip(128) = 256  → 64
        dec1 : up(64)   + skip(64)  = 128  → 32

    Args:
        out_channels   : Output mask channels (default: 1 for binary).
        pretrained     : Load ImageNet pretrained VGG16 (default: True).
        freeze_encoder : Freeze encoder weights (default: False).

    Input:  (B, 3, H, W) — must be ImageNet normalized
    Output: (B, 1, H, W) — values in [0, 1]
    """

    def __init__(
        self,
        out_channels: int    = 1,
        pretrained: bool     = True,
        freeze_encoder: bool = False,
    ):
        super(VGGUNet, self).__init__()

        # ── VGG16 Pretrained Encoder ──────────────────────────
        vgg   = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        feats = list(vgg.features.children())

        # Extract conv blocks — MaxPool layers excluded
        # Pool is applied manually in forward() for skip connection control
        self.enc1 = nn.Sequential(*feats[0:4])  # 3   → 64ch
        self.enc2 = nn.Sequential(*feats[5:9])  # 64  → 128ch
        self.enc3 = nn.Sequential(*feats[10:16])  # 128 → 256ch  ← 17 ki jagah 16
        self.enc4 = nn.Sequential(*feats[17:23])  # 256 → 512ch  ← 18:24 ki jagah 17:23
        self.enc5 = nn.Sequential(*feats[24:30])  # 512 → 512ch  ← 25:31 ki jagah 24:30

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Optionally freeze encoder for ablation experiments
        if freeze_encoder:
            for param in list(self.enc1.parameters()) + \
                         list(self.enc2.parameters()) + \
                         list(self.enc3.parameters()) + \
                         list(self.enc4.parameters()) + \
                         list(self.enc5.parameters()):
                param.requires_grad = False

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = decoder_block(512, 1024)

        # ── Decoder ───────────────────────────────────────────
        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = decoder_block(512 + 512, 512)   # 1024 → 512

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = decoder_block(256 + 512, 256)   # 768  → 256

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = decoder_block(128 + 256, 128)   # 384  → 128

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = decoder_block(64 + 128, 64)     # 192  → 64

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = decoder_block(32 + 64, 32)      # 96   → 32

        # ── Output Head ───────────────────────────────────────
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG16 encoder and UNet decoder.

        Args:
            x: ImageNet-normalized input (B, 3, H, W).

        Returns:
            Sigmoid-activated segmentation mask (B, 1, H, W).
        """
        # ── Encoder ───────────────────────────────────────────
        e1 = self.enc1(x)              # (B, 64,  H,    W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8,  W/8)
        e5 = self.enc5(self.pool(e4))  # (B, 512, H/16, W/16)

        # ── Bottleneck ────────────────────────────────────────
        b  = self.bottleneck(self.pool(e5))  # (B, 1024, H/32, W/32)

        # ── Decoder + Skip Connections ────────────────────────
        d5 = self.dec5(torch.cat([self.up5(b),  e5], dim=1))  # (B, 512, H/16, W/16)
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))  # (B, 256, H/8,  W/8)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, H/4,  W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64,  H/2,  W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32,  H,    W)

        # ── Output ────────────────────────────────────────────
        return torch.sigmoid(self.output(d1))  # (B, 1, H, W)


if __name__ == "__main__":
    # Sanity check — verify output shapes and parameter count
    model  = VGGUNet(pretrained=True)
    x      = torch.randn(2, 3, 256, 256)
    output = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output range : [{output.min():.4f}, {output.max():.4f}]")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params  : {total:,}")
    print(f"Trainable     : {trainable:,}")
