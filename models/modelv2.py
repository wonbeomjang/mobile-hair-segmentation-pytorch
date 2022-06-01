import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2

from .blocks import LayerDepwiseDecode


class MobileHairNetV2(nn.Module):
    def __init__(self, decode_block=LayerDepwiseDecode, *args, **kwargs):
        super(MobileHairNetV2, self).__init__()
        self.mobilenet = mobilenet_v2(*args, **kwargs)
        self.decode_block = decode_block
        self.make_layers()
        self._init_weight()

    def make_layers(self):
        self.encode_layer1 = nn.Sequential(*list(self.mobilenet.features)[:2])
        self.encode_layer2 = nn.Sequential(*list(self.mobilenet.features)[2:4])
        self.encode_layer3 = nn.Sequential(*list(self.mobilenet.features)[4:7])
        self.encode_layer4 = nn.Sequential(*list(self.mobilenet.features)[7:14])
        self.encode_layer5 = nn.Sequential(*list(self.mobilenet.features)[14:19])

        self.decode_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
        )
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=64, kernel_size=1),
            self.decode_block(in_channel=64, out_channel=64, kernel_size=3),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer3 = nn.Sequential(
            self.decode_block(in_channel=64, out_channel=64, kernel_size=3),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer4 = nn.Sequential(
            self.decode_block(in_channel=64, out_channel=64, kernel_size=3),
            nn.Upsample(scale_factor=2)
        )
        self.decode_layer5 = nn.Sequential(
            self.decode_block(in_channel=64, out_channel=64, kernel_size=3),
            nn.Upsample(scale_factor=2),
            self.decode_block(in_channel=64, out_channel=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1)
        )
        
        self.encode_to_decoder4 = nn.Conv2d(in_channels=96, out_channels=1280, kernel_size=1)
        self.encode_to_decoder3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.encode_to_decoder2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=1)
        self.encode_to_decoder1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self._forward_implement(x)
        
        return x
    
    def _forward_implement(self, x):
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
        out = self.decode_layer5(decode_layer4)

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
