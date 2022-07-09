import torch
from torch import nn
import math
import os


class LambdaLR:
    def __init__(self, n_epoch, offset, total_batch_size, decay_batch_size):
        self.n_epoch = n_epoch
        self.offset = offset
        self.total_batch_size = total_batch_size
        self.decay_batch_size = decay_batch_size

    def step(self, epoch):
        factor = pow(0.1, int(((self.offset + epoch) * self.total_batch_size) / self.decay_batch_size))
        return factor


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if math.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quantize_model(net: nn.Module) -> nn.Module:
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net.eval()
    net.fuse_model()
    net.train()
    net = torch.quantization.prepare_qat(net)
    net = torch.quantization.convert(net)

    return net


def get_model_size(net: nn.Module) -> float:
    """
    Get model size
    input:
        net: nn.Module -> deep learning model
    output:
        float: model parameter size calculated by MB.
    """
    torch.save(net.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    return model_size

