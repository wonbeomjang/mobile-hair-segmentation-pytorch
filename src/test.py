import time

import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image
from utils.custom_transfrom import UnNormalize
from utils.util import AverageMeter, get_model_size
from loss.loss import iou_loss
from models import *


class Tester:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.num_classes = config.num_classes
        self.num_test = config.num_test
        self.sample_dir = config.sample_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.quantize = config.quantize
        self.model_version = config.model_version
        if self.quantize:
            self.device = torch.device('cpu')
        self.build_model()

    def build_model(self):
        if self.model_version == 1:
            if self.quantize:
                self.net = quantized_modelv1(pretrained=True).to(self.device)
            else:
                self.net = modelv1(pretrained=True).to(self.device)
        elif self.model_version == 2:
            if self.quantize:
                self.net = quantized_modelv2(pretrained=True).to(self.device)
            else:
                self.net = modelv2(pretrained=True).to(self.device)
        else:
            raise Exception('[!] Unexpected model version')
        self.load_model()
        
    def load_model(self):
        ckpt = f'{self.checkpoint_dir}/quantized.pth' if self.quantize else f'{self.checkpoint_dir}/best.pth'
        if not os.path.exists(ckpt):
            return
        print(f'[*] Load Model from {ckpt}')
        save_info = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(save_info['state_dict'])

    def test(self, net=None):
        if net:
            self.net = net
        self.net = self.net.eval()
        avg_meter = AverageMeter()
        inference_avg = AverageMeter()
        with torch.no_grad():
            unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            model_size = get_model_size(self.net)
            for step, (image, mask) in pbar:
                image = image.to(self.device)
                #image = unnormal(image.to(self.device))
                cur = time.time()
                result = self.net(image)
                inference_avg.update(time.time() - cur)

                mask = mask.to(self.device)

                avg_meter.update(iou_loss(result, mask))
                pbar.set_description(f"IOU: {avg_meter.avg:.4f} | "
                                     f"Model Size: {model_size:.2f}MB | Infernece Speed: {inference_avg.avg:.4f}")

                mask = mask.repeat_interleave(3, 1)
                argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
                result = result[:, 1, :, :].unsqueeze(dim=1)
                result = result * argmax
                result = result.repeat_interleave(3, 1)
                torch.cat([image, result, mask])

                save_image(torch.cat([image, result, mask]), os.path.join(self.sample_dir, f"{step}.png"))

