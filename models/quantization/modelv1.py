import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional

from torchvision.models.quantization.mobilenet import mobilenet_v2
from torchvision.models.quantization.utils import _replace_relu
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torchvision.models.quantization.mobilenetv2 import QuantizableInvertedResidual
from torchvision.ops.misc import ConvNormActivation

from ..modelv1 import MobileHairNet
from .blocks import QuantizableLayerDepwiseDecode, QuantizableLayerDepwiseEncode, QuantizeUpSample


class QuantizableMobileHairNet(MobileHairNet):
    def __init__(self):
        super(QuantizableMobileHairNet, self).__init__(encode_block=QuantizableLayerDepwiseEncode, decode_block=QuantizableLayerDepwiseDecode)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.f_add = FloatFunctional()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_implement(x)
        x = self.dequant(x)
        
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

        decode_layer1 = self.f_add.add(self.decode_layer1(encode_layer5), encode_layer4)
        decode_layer2 = self.f_add.add(self.decode_layer2(decode_layer1), encode_layer3)
        decode_layer3 = self.f_add.add(self.decode_layer3(decode_layer2), encode_layer2)
        decode_layer4 = self.f_add.add(self.decode_layer4(decode_layer3), encode_layer1)
        decode_layer5 = self.decode_layer5(decode_layer4)

        out = decode_layer5
        return out

    def fuse_model(self) -> None:
        for m in self.modules():
            if type(m) is ConvNormActivation:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) is QuantizableInvertedResidual or type(m) is QuantizableLayerDepwiseDecode:
                m.fuse_model()

    def quantize(self) -> None:
        self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.eval()
        self.fuse_model()
        self.train()
        torch.quantization.prepare_qat(self, inplace=True)
        torch.quantization.convert(self, inplace=True)