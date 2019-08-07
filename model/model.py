import torch
import torch.nn as nn
from config.config import get_config
from torch.nn.init import xavier_normal_

config = get_config()


class _Layer_Depwise_Encode(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, reserve=False): #nf==64
        self.stride = int(out_channels/in_channels)
        if reserve == True:
            self.stride = 1
        super(_Layer_Depwise_Encode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class _Layer_Depwise_Decode(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,  stride=1):
        super(_Layer_Depwise_Decode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class MobileHairNet(nn.Module):
    def __init__(self, im_size=224, nf=32, kernel_size=3):
        super(MobileHairNet, self).__init__()
        self.nf = nf
#######################################################################################################################
#                                                                                                                     #
#                                               ENCODER                                                               #
#                                                                                                                     #
#######################################################################################################################
        self.encode_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=kernel_size, stride=2, padding=1),
            _Layer_Depwise_Encode(nf, 2*nf, reserve=True)
        )
        self.encode_layer2 = nn.Sequential(
            _Layer_Depwise_Encode(2*nf, 4*nf),
            _Layer_Depwise_Encode(4*nf, 4*nf),
        )
        self.encode_layer3 = nn.Sequential(
            _Layer_Depwise_Encode(4*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf)
        )
        self.encode_layer4 = nn.Sequential(
            _Layer_Depwise_Encode(8*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
        )
        self.encode_layer5 = nn.Sequential(
            _Layer_Depwise_Encode(16*nf, 32*nf),
            _Layer_Depwise_Encode(32*nf, 32*nf)
        )
#######################################################################################################################
#                                                                                                                     #
#                                               DECODER                                                               #
#                                                                                                                     #
#######################################################################################################################
        self.decode_layer1 = nn.Upsample(scale_factor=2)
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32*nf, out_channels=2*nf, kernel_size=1),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer3 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer4 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer5 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Conv2d(in_channels=2*nf, out_channels=2, kernel_size=kernel_size, padding=1)
        )
        self.encode_to_decoder4 = nn.Conv2d(in_channels=16 * self.nf, out_channels=32 * self.nf, kernel_size=1)
        self.encode_to_decoder3 = nn.Conv2d(in_channels=8 * self.nf, out_channels=2 * self.nf, kernel_size=1)
        self.encode_to_decoder2 = nn.Conv2d(in_channels=4 * self.nf, out_channels=2 * self.nf, kernel_size=1)
        self.encode_to_decoder1 = nn.Conv2d(in_channels=2 * self.nf, out_channels=2 * self.nf, kernel_size=1)
        self.soft_max = nn.Softmax(dim=1)
    def forward(self, x):
#connet encode 4-> decode 1, encode 3-> decode 2, encode 2-> decode 3, encode 1-> decode 4
        encode_layer1 = self.encode_layer1(x)
        encode_layer2 = self.encode_layer2(encode_layer1)
        encode_layer3 = self.encode_layer3(encode_layer2)
        encode_layer4 = self.encode_layer4(encode_layer3)
        encode_layer5 = self.encode_layer5(encode_layer4)

        encode_layer4 = self.encode_to_decoder4(encode_layer4)
        encode_layer3 = self.encode_to_decoder3(encode_layer3)
        encode_layer2 = self.encode_to_decoder2(encode_layer2)
        encode_layer1 = self.encode_to_decoder1(encode_layer1)

        decode_layer1 = torch.add(self.decode_layer1(encode_layer5), encode_layer4)
        decode_layer2 = torch.add(self.decode_layer2(decode_layer1), encode_layer3)
        decode_layer3 = torch.add(self.decode_layer3(decode_layer2), encode_layer2)
        decode_layer4 = torch.add(self.decode_layer4(decode_layer3), encode_layer1)
        decode_layer5 = self.decode_layer5(decode_layer4)
        out = self.soft_max(decode_layer5)
        return out

    def _init_weight(self):
        layers = [self.encode_layer1, self.encode_layer2, self.encode_layer3, self.encode_layer4, self.encode_layer5,
                  self.decode_layer1, self.decode_layer2, self.decode_layer3, self.decode_layer4, self.decode_layer5]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_normal_(layer.weight.data)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)