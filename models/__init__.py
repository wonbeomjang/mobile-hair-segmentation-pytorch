import torch.utils.model_zoo

from .modelv1 import *
from .modelv2 import *
from .quantization.modelv1 import *
from .quantization.modelv2 import *

url_map = {
    'hairmattenetv1': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/hairmattenet_v1.pth',
    'hairmattenetv2': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/hairmattenet_v2.pth',
    'quantized_hairmattenetv1': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/quantized_hairmattenetv1-a32fsc.pth',
    'quantized_hairmattenetv2': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/quantized_hairmattenetv2-a32fsc.pth'}

model_map = {'hairmattenetv1': MobileHairNet,
             'hairmattenetv2': MobileHairNetV2,
             'quantized_hairmattenetv1': QuantizableMobileHairNet,
             'quantized_hairmattenetv2': QuantizableMobileHairNetV2}


def _model(arch: str, pretrained: bool, quantize=False):
    print(f"[*] Load {arch}")
    model: nn.Module = model_map[arch]()

    if pretrained:
        state_dict = torch.utils.model_zoo.load_url(url_map[arch])
        if quantize:
            model.quantize()
        model.load_state_dict(state_dict)

    return model


def modelv1(pretrained=False):
    return _model('hairmattenetv1', pretrained)


def modelv2(pretrained=False):
    return _model('hairmattenetv2', pretrained)


def quantized_modelv1(pretrained=False):
    return _model('quantized_hairmattenetv1', pretrained, True)


def quantized_modelv2(pretrained=False):
    return _model('quantized_hairmattenetv2', pretrained, True)
