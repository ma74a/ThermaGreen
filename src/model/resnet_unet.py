import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_connection, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels+skip_connection, out_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None:
            # Padding in case the input dimensions are not perfectly divisible by 2
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            
            # Concatenate along the channel dimension
            x = torch.cat([skip, x], dim=1)

        return self.conv(x)



class ResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNetUNet, self).__init__()
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.pool = model.maxpool
        self.encoder1 = model.layer1
        self.encoder2 = model.layer2
        self.encoder3 = model.layer3
        self.encoder4 = model.layer4

        # Decoder
        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128,  64,  64)
        self.up1 = DecoderBlock( 64,  64,  64)
        self.up0 = DecoderBlock(64, 0, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3) # Bottleneck layer

        d = self.up4(e4, skip=e3)
        d = self.up3(d, skip=e2)
        d = self.up2(d, skip=e1)
        d = self.up1(d, skip=e0)
        d = self.up0(d)

        return self.final_conv(d)
