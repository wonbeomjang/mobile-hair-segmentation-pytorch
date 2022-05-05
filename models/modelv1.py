import torch.nn as nn

from .blocks import LayerDepwiseDecode, LayerDepwiseEncode


class MobileHairNet(nn.Module):
    def __init__(self, encode_block=LayerDepwiseEncode, decode_block=LayerDepwiseDecode, *args, **kwargs):
        super(MobileHairNet, self).__init__()
        self.encode_block = encode_block
        self.decode_block = decode_block
        self.make_layers()
        
    def make_layers(self): 
        self.encode_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            self.encode_block(32, 64, reserve=True)
        )
        self.encode_layer2 = nn.Sequential(
            self.encode_block(64, 128),
            self.encode_block(128, 128),
        )
        self.encode_layer3 = nn.Sequential(
            self.encode_block(128, 256),
            self.encode_block(256,256)
        )
        self.encode_layer4 = nn.Sequential(
            self.encode_block(256, 512),
            self.encode_block(512, 512),
            self.encode_block(512, 512),
            self.encode_block(512, 512),
            self.encode_block(512, 512),
            self.encode_block(512, 512),
        )
        self.encode_layer5 = nn.Sequential(
            self.encode_block(512, 1024),
            self.encode_block(1024, 1024)
        )
        
        self.decode_layer1 = nn.Upsample(scale_factor=2)
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1),
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
        self.encode_to_decoder4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.encode_to_decoder3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.encode_to_decoder2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.encode_to_decoder1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.soft_max = nn.Softmax(dim=1)

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