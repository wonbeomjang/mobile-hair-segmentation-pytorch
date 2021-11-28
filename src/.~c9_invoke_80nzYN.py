import os
from glob import glob

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.adadelta import Adadelta

from models.modelv1 import MobileHairNet
from models.modelv2 import MobileHairNetV2
from loss.loss import ImageGradientLoss, iou_loss
from utils.util import LambdaLR, AverageMeter

class Trainer:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = 0
        self.num_epoch = config.num_epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        self.gradient_loss_weight = config.gradient_loss_weight
        self.model_version = config.model_version

        self.build_model()
        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=config.eps, rho=config.rho, weight_decay=config.decay)

    def build_model(self):
        if self.model_version == 1:
            self.net = MobileHairNet().to(self.device)
        elif self.model_version == 2:
            self.net = MobileHairNetV2().to(self.device)
            
        else:
            raise Exception('[!] Unexpected model version')
            
        self.load_model()

    def load_model(self):
        
        if not self.model_path:
            return
        
        save_info = torch.load(self.model_path)
        # save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        
        self.epoch = save_info['epoch']
        self.net = save_info['model']
        self.net.load_state_dict(save_info['state_dict'], map_location=self.device)
        self.optimizer = save_info['optimizer']
        
        print(f"[*] Load Model from {self.model_path}")

    def train(self):
        bce_losses = AverageMeter()
        image_gradient_losses = AverageMeter()
        image_gradient_criterion = ImageGradientLoss().to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epoch, self.num_epoch):
            bce_losses.reset()
            image_gradient_losses.reset()
            
            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            for step, (image, gray_image, mask) in pbar:
                image = image.to(self.device)
                mask = mask.to(self.device)
                gray_image = gray_image.to(self.device)

                pred = self.net(image)

                pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()

                # preds_flat.shape (N*224*224, 2)
                # masks_flat.shape (N*224*224, 1)
                image_gradient_loss = image_gradient_criterion(pred, gray_image)
                bce_loss = bce_criterion(pred_flat, mask_flat)

                loss = bce_loss + self.gradient_loss_weight * image_gradient_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bce_losses.update(bce_loss.item(), self.batch_size)
                image_gradient_losses.update(self.gradient_loss_weight * image_gradient_loss, self.batch_size)
                iou = iou_loss(pred, mask)

                # save sample images
                pbar.set_description(f"Epoch: [{epoch}/{self.num_epoch}] | Bce Loss: {bce_losses.avg:.4f} | "
                    f"Image Gradient Loss: {image_gradient_losses.avg:.4f} | IOU: {iou:.4f}")
                
                if step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, step)
                    print('[*] Saved sample images')
            
            save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(save_info, f'{self.checkpoint_dir}/last.pth')

    def save_sample_imgs(self, real_img, real_mask, prediction, save_dir, epoch, step):
        data = [real_img, real_mask, prediction]
        names = ["Image", "Mask", "Prediction"]

        fig = plt.figure()
        for i, d in enumerate(data):
            d = d.squeeze()
            im = d.data.cpu().numpy()

            if i > 0:
                im = np.expand_dims(im, axis=0)
                im = np.concatenate((im, im, im), axis=0)

            im = (im.transpose(1, 2, 0) + 1) / 2

            f = fig.add_subplot(1, 3, i + 1)
            f.imshow(im)
            f.set_title(names[i])
            f.set_xticks([])
            f.set_yticks([])

        p = os.path.join(save_dir, "epoch-%s_step-%s.png" % (epoch, step))
        plt.savefig(p)