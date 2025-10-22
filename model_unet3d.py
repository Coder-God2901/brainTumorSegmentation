# model_unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, n_classes=4, base_c=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_c)
        self.enc2 = DoubleConv(base_c, base_c*2)
        self.enc3 = DoubleConv(base_c*2, base_c*4)
        self.enc4 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(base_c*8, base_c*16)
        self.up4 = nn.ConvTranspose3d(base_c*16, base_c*8, 2, stride=2)
        self.dec4 = DoubleConv(base_c*16, base_c*8)
        self.up3 = nn.ConvTranspose3d(base_c*8, base_c*4, 2, stride=2)
        self.dec3 = DoubleConv(base_c*8, base_c*4)
        self.up2 = nn.ConvTranspose3d(base_c*4, base_c*2, 2, stride=2)
        self.dec2 = DoubleConv(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose3d(base_c*2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c*2, base_c)
        self.out_conv = nn.Conv3d(base_c, n_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        d4 = self.dec4(torch.cat([self.up4(b), c4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.out_conv(d1)
