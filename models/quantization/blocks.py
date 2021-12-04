import torch.nn as nn
from torch.quantization import fuse_modules

class QuantizableLayerDepwiseEncode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reserve=False):
        self.stride = int(out_channels/in_channels)
        if reserve == True:
            self.stride = 1
        super(QuantizableLayerDepwiseEncode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out
        
    def fuse_model(self):
        fuse_modules(self.layer, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer, ["3", "4", "5"], inplace=True)
        

class QuantizableLayerDepwiseDecode(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,  stride=1):
        super(QuantizableLayerDepwiseDecode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out
        
    def fuse_model(self):
        fuse_modules(self.layer, [str(1), str(2)], inplace=True)