import torch
import time
from tqdm import tqdm
import torch_tensorrt

import data
from models import *
from config.config import get_config
from utils.util import get_model_size, AverageMeter
from loss.loss import iou_loss


def convert_tensrrt(net):
    trt_ts_module = torch_tensorrt.compile(torch.jit.script(net),
                                           inputs=[torch_tensorrt.Input([1, 3, 224, 224], dtype=torch.float32)],
                                           enabled_precisions={torch.float32},
                                           )

    return trt_ts_module


def test_model(net, dataloader):
    net = net.eval()
    pbar = tqdm(dataloader)

    avg_meter = AverageMeter()
    inference_avg = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        model_size = get_model_size(net)
        for step, (image, mask) in pbar:
            image = image.to(device)
            cur = time.time()
            result = net(image)
            inference_avg.update(time.time() - cur)

            mask = mask.to(device)

            avg_meter.update(iou_loss(result, mask))
            pbar.set_description(f"IOU: {avg_meter.avg:.4f} | "
                                 f"Model Size: {model_size:.2f}MB | Infernece Speed: {inference_avg.avg:.4f}")

    return model_size, inference_avg.avg


if __name__ == "__main__":
    device = torch.device("cuda:0")
    config = get_config()

    net = get_model("hairmattenetv1").to(device)
    trt_ts_module = convert_tensrrt(net)

    test_loader = data.test_loader.get_loader(config.test_data_path, config.test_batch_size, config.image_size,
                                              shuffle=None, num_workers=int(config.workers))
    torch.jit.save(trt_ts_module, "trt_torchscript_module.ts")

