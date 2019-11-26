import torch.nn as nn
from config.config import get_config
from torchvision.models.mobilenet import mobilenet_v2

config = get_config()

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
    def __init__(self, kernel_size=3, pretrained=True):
        super(MobileHairNet, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
#######################################################################################################################
#                                                                                                                     #
#                                               ENCODER                                                               #
#                                                                                                                     #
#######################################################################################################################
        self.encode_layer1 = nn.Sequential(*list(mobilenet.features)[:2])
        self.encode_layer2 = nn.Sequential(*list(mobilenet.features)[2:4])
        self.encode_layer3 = nn.Sequential(*list(mobilenet.features)[4:7])
        self.encode_layer4 = nn.Sequential(*list(mobilenet.features)[7:14])
        self.encode_layer5 = nn.Sequential(*list(mobilenet.features)[14:19])

#######################################################################################################################
#                                                                                                                     #
#                                               DECODER                                                               #
#                                                                                                                     #
#######################################################################################################################
        self.decode_layer1 = nn.Upsample(scale_factor=2)
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=64, kernel_size=1),
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer3 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer4 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer5 = nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=kernel_size, padding=1)
        )
        self.encode_to_decoder4 = nn.Conv2d(in_channels=96, out_channels=1280, kernel_size=1)
        self.encode_to_decoder3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.encode_to_decoder2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=1)
        self.encode_to_decoder1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1)

        self.soft_max = nn.Softmax(dim=1)

        if pretrained:
            self._init_weight()

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

        decode_layer1 = self.decode_layer1(encode_layer5) + encode_layer4
        decode_layer2 = self.decode_layer2(decode_layer1) + encode_layer3
        decode_layer3 = self.decode_layer3(decode_layer2) + encode_layer2
        decode_layer4 = self.decode_layer4(decode_layer3) + encode_layer1
        decode_layer5 = self.decode_layer5(decode_layer4)

        out = self.soft_max(decode_layer5)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)