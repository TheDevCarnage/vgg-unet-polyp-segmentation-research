import torch
import torch.nn as nn


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.
    This is the fundamental building block of UNet at every level.

    Args:
        in_channels  : Number of input feature channels.
        out_channels : Number of output feature channels.

    Returns:
        Sequential block with two conv layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetBaseline(nn.Module):
    """
    Standard U-Net architecture for binary image segmentation.

    This serves as the baseline model against which the VGG16-UNet
    hybrid will be compared. Encoder is randomly initialized —
    no pretrained weights.

    Architecture:
        Encoder: 4 downsampling blocks (MaxPool + double conv)
        Bottleneck: double conv at lowest resolution
        Decoder: 4 upsampling blocks (ConvTranspose2d + skip + double conv)
        Output: 1x1 Conv → Sigmoid for binary mask

    Args:
        in_channels  : Input image channels (default: 3 for RGB).
        out_channels : Output mask channels (default: 1 for binary).
        features     : Channel progression per encoder level.

    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) — values in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
    ):
        super(UNetBaseline, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Encoder ───────────────────────────────────────────
        self.enc1 = double_conv(in_channels, features[0])  # 3   → 64
        self.enc2 = double_conv(features[0], features[1])  # 64  → 128
        self.enc3 = double_conv(features[1], features[2])  # 128 → 256
        self.enc4 = double_conv(features[2], features[3])  # 256 → 512

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = double_conv(features[3], features[3] * 2)  # 512 → 1024

        # ── Decoder ───────────────────────────────────────────
        # Each decoder step: upsample + concatenate skip + double conv
        self.up4 = nn.ConvTranspose2d(
            features[3] * 2, features[3], kernel_size=2, stride=2
        )
        self.dec4 = double_conv(features[3] * 2, features[3])  # 1024 → 512

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = double_conv(features[2] * 2, features[2])  # 512  → 256

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = double_conv(features[1] * 2, features[1])  # 256  → 128

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = double_conv(features[0] * 2, features[0])  # 128  → 64

        # ── Output ────────────────────────────────────────────
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.

        Skip connections preserve spatial information from encoder
        to decoder at each resolution level.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Binary segmentation mask of shape (B, 1, H, W).
        """
        # ── Encoder ───────────────────────────────────────────
        e1 = self.enc1(x)  # (B, 64,  H,    W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8,  W/8)

        # ── Bottleneck ────────────────────────────────────────
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # ── Decoder + skip connections ────────────────────────
        d4 = self.up4(b)  # (B, 512, H/8,  W/8)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # (B, 512, H/8,  W/8)

        d3 = self.up3(d4)  # (B, 256, H/4,  W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, 256, H/4,  W/4)

        d2 = self.up2(d3)  # (B, 128, H/2,  W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 128, H/2,  W/2)

        d1 = self.up1(d2)  # (B, 64,  H,    W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, 64,  H,    W)

        # ── Output ────────────────────────────────────────────
        return torch.sigmoid(self.output(d1))  # (B, 1,   H,    W)


if __name__ == "__main__":
    # Quick sanity check — verify shapes are correct
    model = UNetBaseline()
    x = torch.randn(2, 3, 256, 256)  # Batch of 2 images
    output = model(x)

    print(f"Input  shape : {x.shape}")  # [2, 3, 256, 256]
    print(f"Output shape : {output.shape}")  # [2, 1, 256, 256]
    print(f"Output range : [{output.min():.4f}, {output.max():.4f}]")  # [0, 1]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")  # ~31M