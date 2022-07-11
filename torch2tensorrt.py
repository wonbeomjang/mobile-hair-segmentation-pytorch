import torch
import time
import os
from tqdm import tqdm
import torch_tensorrt

import data.test_loader
from models import *
from config.config import get_config
from utils.util import get_model_size, AverageMeter
from loss.loss import iou_loss


def build_model(config):
    if config.model_version == 1:
        net = modelv1(pretrained=True)
    elif config.model_version == 2:
        net = modelv2(pretrained=True)
    else:
        raise Exception('[!] Unexpected model version')
    return net


def load_model(net, config, device):
    ckpt = config.model_path

    if not ckpt:
        return net
    print(f'[*] Load Model from {ckpt}')
    save_info = torch.load(ckpt, map_location=device)
    if config.quantize:
        net.quantize()
    net.load_state_dict(save_info['state_dict'])

    return net


def convert_tensrrt(net):
    trace_model = torch.jit.script(net)
    trt_ts_module = torch_tensorrt.compile(trace_model,
                                           inputs=[torch_tensorrt.Input([1, 3, 224, 224], dtype=torch.float32)],
                                           enabled_precisions={torch.float32},
                                           )
    return trt_ts_module


def test_model(net, dataloader, device):
    net = net.eval()

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
                                 f"Model Size: {model_size * 1000:.4f}KB | Infernece Speed: {inference_avg.avg:.4f}")

    return model_size, inference_avg.avg


if __name__ == "__main__":
    device = torch.device("cuda:0")
    config = get_config()

    net = build_model(config).to(device).eval()
    net = load_model(net, config, device)
    trt_ts_module = convert_tensrrt(net)
    
    if os.path.exists(config.test_data_path):
        test_loader = data.test_loader.get_loader(config.test_data_path, config.test_batch_size, config.image_size,
                                                  shuffle=None, num_workers=int(config.workers))
        print("[*] Before deploy TensorRT")
        test_model(net, test_loader, device)
        print("[*] After deploy TensorRT")
        test_model(trt_ts_module, test_loader, device)
    torch.jit.save(trt_ts_module, "trt_torchscript_module.ts")


