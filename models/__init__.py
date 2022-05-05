import torch.utils.model_zoo
import torch.nn as nn
import torch

from .modelv1 import *
from .modelv2 import *
from .quantization.modelv1 import *
from .quantization.modelv2 import *

url_map = {
    'hairmattenetv1': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/modelv1.pt',
    'hairmattenetv2': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/modelv2.pt',
    'quantized_hairmattenetv1': None,
    'quantized_hairmattenetv2': 'https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch/releases/download/paramter/quantized_modelv2.pt'}

model_map = {'hairmattenetv1': MobileHairNet,
             'hairmattenetv2': MobileHairNetV2,
             'quantized_hairmattenetv1': QuantizableMobileHairNet,
             'quantized_hairmattenetv2': QuantizableMobileHairNetV2}


def _model(arch: str, train: bool, pretrained: bool, device=None):
    print(f"[*] Load {arch}")
    device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = None
    if pretrained:
        if not url_map[arch]:
            print(f"[!]{arch} is not ready")
            model: nn.Module = model_map[arch]()
        else:
            state_dict = torch.utils.model_zoo.load_url(url_map[arch], map_location=device)
            if arch.startswith('quantized'):
                model = torch.jit.load(state_dict)
            else:
                model: nn.Module = state_dict['model']
                model.load_state_dict(state_dict['state_dict'])
    else:
        model: nn.Module = model_map[arch]()

    if state_dict and train:
        return model, state_dict
    return model


def modelv1(train=False, pretrained=False, device=None):
    return _model('hairmattenetv1', train, pretrained, device)


def modelv2(train=False, pretrained=False, device=None):
    return _model('hairmattenetv2', train, pretrained, device)


def quantized_modelv1(train=False, pretrained=False, device=None):
    return _model('quantized_hairmattenetv1', train, pretrained, device)


def quantized_modelv2(train=False, pretrained=False, device=None):
    return _model('quantized_hairmattenetv2', train, pretrained, device)
