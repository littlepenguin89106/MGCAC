"""
Counter modules.
"""
import torch
from torch import nn
import torch.nn.functional as F

# The code of double convolution block is from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3,1,1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def get_counter(cfg):
    return DensityHead()

class DensityHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_x3 = nn.Conv2d(384,256,1)
        self.conv_x2 = nn.Conv2d(192,128,1)
        self.conv_x1 = nn.Conv2d(64,64,1)


        self.up3 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.up2 = Up(128+128,64)
        self.up1 = Up(64+64,64)
        self.up0 = nn.Sequential(
            nn.Conv2d(64,32,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,1),
            nn.ReLU(inplace=True),
        )
        
        self._weight_init_()
        
    def forward(self, x):
        x3 = self.conv_x3(x[2])
        x2 = self.conv_x2(x[1])
        x1 = self.conv_x1(x[0])
        d3 = self.up3(x3)
        d2 = self.up2(d3,x2)
        d1 = self.up1(d2,x1)
        x = self.up0(d1)

        return x
        
    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
