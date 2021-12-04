import torch
from torch import nn

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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quantize_model(model: nn.Module, backend: str) -> None:
    _dummy_input_data = torch.rand(1, 3, 224, 224)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == "fbgemm":
        model.qconfig = torch.ao.quantization.qconfig.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.observer.default_observer,
            weight=torch.ao.quantization.observer.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        model.qconfig = torch.ao.quantization.qconfig.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.observer.default_observer, weight=torch.ao.quantization.default_weight_observer
        )

    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    model.fuse_model()  # type: ignore[operator]
    torch.ao.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.ao.quantization.convert(model, inplace=True)

    return
