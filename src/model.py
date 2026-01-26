# src/model.py
"""
U-Net model for Brain MRI Tumor Segmentation.
Author: Shruti Suman
Description:
    This file defines the U-Net architecture in PyTorch.
    It takes grayscale MRI images (1 channel) and predicts binary tumor masks (1 channel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 🔹 Basic building block: Double Convolution (Conv → BN → ReLU) × 2
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# ----------------------------------------------------------------------
# 🔹 U-Net Architecture
# ----------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        """
        Args:
            n_channels: Input channels (1 for grayscale MRI)
            n_classes: Output channels (1 for binary tumor mask)
        """
        super(UNet, self).__init__()

        # Encoder (Downsampling Path)
        self.conv1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (Deepest Layer)
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Upsampling Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final Output Layer (1 channel = mask)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through U-Net.
        Input:  (B, 1, H, W)
        Output: (B, 1, H, W) - segmentation mask with pixel values in [0, 1]
        """
        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        # Bottleneck
        x9 = self.bottleneck(x8)

        # Decoder
        x = self.upconv4(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        # Final 1×1 Conv and Sigmoid activation
        x = self.final_conv(x)
        return torch.sigmoid(x)


# ----------------------------------------------------------------------
# ✅ Quick Sanity Check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn(1, 1, 256, 256)  # Dummy input (batch=1)
    y = model(x)
    print("✅ Model check successful!")
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
