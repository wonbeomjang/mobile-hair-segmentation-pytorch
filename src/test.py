import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image
from utils.custom_transfrom import UnNormalize
from utils.util import AverageMeter
from loss.loss import iou_loss


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
        ckpt = f'{self.checkpoint_dir}/quantized.pt' if self.quantize else f'{self.checkpoint_dir}/best.pth'
        print(f'[*] Load Model from {ckpt}')

        if self.quantize:
            self.net = torch.jit.load(ckpt, map_location=self.device)
        else:
            save_info = torch.load(ckpt, map_location=self.device)
            self.net = save_info['model']
            self.net.load_state_dict(save_info['state_dict'])

    def test(self, net=None):
        if net:
            self.net = net
        self.net = self.net.eval()
        avg_meter = AverageMeter()
        with torch.no_grad():
            unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            for step, (image, mask) in pbar:
                image = image.to(self.device)
                #image = unnormal(image.to(self.device))
                result = self.net(image)

                mask = mask.to(self.device)

                avg_meter.update(iou_loss(result, mask))
                pbar.set_description(f'IOU: {avg_meter.avg:.4f}')

                mask = mask.repeat_interleave(3, 1)
                argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
                result = result[:, 1, :, :].unsqueeze(dim=1)
                result = result * argmax
                result = result.repeat_interleave(3, 1)
                torch.cat([image, result, mask])


                save_image(torch.cat([image, result, mask]), os.path.join(self.sample_dir, f"{step}.png"))

