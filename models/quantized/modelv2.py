import torch
import torch.nn as nn

from torchvision.models.quantization.mobilenet import mobilenet_v2
from torchvision.models.quantization.utils import _replace_relu
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torchvision.models.quantization.mobilenetv2 import QuantizableInvertedResidual
from torchvision.ops.misc import ConvNormActivation

from model.modelv2 import MobileHairNetV2

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


class QuantizableMobileHairNet(MobileHairNetV2):
    def __init__(self, decode_block=QuantizableLayerDepwiseDecode, *args, **kwargs):
        super(QuantizableMobileHairNet, self).__init__()
        
        self.mobilenet = mobilenet_v2()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        _replace_relu(self.mobilenet)
        self.make_layers()
        
    def forward(self, x):
        x = self.quant(x)
        x = self._forward_implement(x)
        x = self.dequant(x)
        
        return x
        
    def fuse_model(self) -> None:
        for m in self.modules():
            if type(m) is ConvNormActivation:
                fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) is QuantizableInvertedResidual and type(m) is QuantizableLayerDepwiseDecode:
                m.fuse_model()

def quantize_model(model: nn.Module, backend: str) -> None:
    _dummy_input_data = torch.rand(1, 3, 224, 224)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == "fbgemm":
        model.qconfig = torch.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        model.qconfig = torch.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.quantization.default_observer, weight=torch.quantization.default_weight_observer
        )

    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    model.fuse_model()  # type: ignore[operator]
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

    return

