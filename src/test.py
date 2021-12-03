import torch
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.custom_transfrom import UnNormalize
from torchvision.models.quantization.utils import quantize_model

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
        if self.quantize:
            self.device = torch.device('cpu')
        self.load_model()
        
    def load_model(self):
        print(self.quantize)
        ckpt = f'{self.checkpoint_dir}/quantized.pt' if self.quantize else f'{self.checkpoint_dir}/last.pt'
        print(f'[*] Load Model from {ckpt}')
        save_info = torch.load(ckpt, map_location=self.device)
        # save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict()}
         
        self.epoch = save_info['epoch']
        self.net = save_info['model']
        self.optimizer = save_info['optimizer']
        print(self.net)

        #if self.quantize:
        #    quantize_model(self.net, "fbgemm")

    def test(self):
        unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for step, (image, mask) in pbar:
            image = unnormal(image.to(self.device))
            mask = mask.to(self.device).repeat_interleave(3, 1)
            result = self.net(image)
            argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
            result = result[:, 1, :, :].unsqueeze(dim=1)
            result = result * argmax
            result = result.repeat_interleave(3, 1)
            torch.cat([image, result, mask])

            save_image(torch.cat([image, result, mask]), os.path.join(self.sample_dir, f"{step}.png"))

