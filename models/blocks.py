import torch.nn as nn


class LayerDepwiseEncode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reserve=False):
        self.stride = int(out_channels/in_channels)
        if reserve == True:
            self.stride = 1
        super(LayerDepwiseEncode, self).__init__()
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


class LayerDepwiseDecode(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,  stride=1):
        super(LayerDepwiseDecode, self).__init__()
        block = [
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
            nn.ReLU(inplace=True)
        ]
        self.layer = nn.Sequential(*block)

    def forward(self, x):
        out = self.layer(x)
        return out