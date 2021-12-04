import os
import time
from glob import glob
import copy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.adadelta import Adadelta

from models.modelv1 import MobileHairNet
from models.modelv2 import MobileHairNetV2
from models.quantization.modelv1 import QuantizableMobileHairNet
from models.quantization.modelv2 import QuantizableMobileHairNetV2
from loss.loss import ImageGradientLoss, iou_loss
from utils.util import LambdaLR, AverageMeter

class Trainer:
    def __init__(self, config, dataloader, val_loader=None):
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = 0
        self.num_epoch = config.num_epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.val_loader = val_loader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        self.gradient_loss_weight = config.gradient_loss_weight
        self.model_version = config.model_version
        self.quantize = config.quantize
        self.resume = config.resume

        self.build_model()
        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=config.eps, rho=config.rho, weight_decay=config.decay)

    def build_model(self):
        if self.model_version == 1:
            if self.quantize:
                self.net = QuantizableMobileHairNet().to(self.device)
            else:
                self.net = MobileHairNet().to(self.device)
        elif self.model_version == 2:
            if self.quantize:
                self.net = QuantizableMobileHairNetV2().to(self.device)
            else:
                self.net = MobileHairNetV2().to(self.device)
            
        else:
            raise Exception('[!] Unexpected model version')
            
        self.load_model()

    def load_model(self):
        if not self.model_path and not self.resume:
            return
        
        ckpt = os.path.join(self.checkpoint_dir, 'last.pt') if self.resume else self.model_path 

        save_info = torch.load(ckpt, map_location=self.device)
        # save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        
        self.epoch = save_info['epoch'] + 1
        self.net = save_info['model']
        self.optimizer = save_info['optimizer']
        self.net.load_state_dict(save_info['state_dict'])
        
        print(f"[*] Load Model from {ckpt}")

    def train(self):
        image_gradient_criterion = ImageGradientLoss().to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)
        best = 0

        if os.path.exists('results.csv'):
            f = open('results.csv', 'a')
        else:
            f = open('results.csv', 'w')
            f.write('epoch,iou,loss\n')

        for epoch in range(self.epoch, self.num_epoch):
            self._train_one_epoch(epoch, image_gradient_criterion, bce_criterion)
            if self.val_loader:
                iou, loss = self.val(image_gradient_criterion, bce_criterion)
                f.write(f'{epoch},{iou:.4f},{loss:4f}\n')
                if iou > best:
                    best = iou
                    save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict(), 'epoch': epoch}
                    torch.save(save_info, f'{self.checkpoint_dir}/best.pt')

        print('Final IOU: {best:.4f}')
        f.close()
    
    def quantize_model(self):
        if not self.quantize:
            return

        print('Load Best Model')
        ckpt = f'{self.checkpoint_dir}/last.pt'
        save_info = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(save_info['state_dict'])
        # save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict()}

       
        print('Before quantize')
        self.device = torch.device('cpu')
        image_gradient_criterion = ImageGradientLoss(device=self.device).to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)

        self.net = self.net.to(self.device)
        self.val(image_gradient_criterion, bce_criterion)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        image_gradient_criterion.device = self.device

        self.net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.net.fuse_model()
        self.net = torch.quantization.prepare_qat(self.net)
        
        self._train_one_epoch(0, image_gradient_criterion, bce_criterion, quantize=True)
        
        self.device = torch.device('cpu')
        self.net = self.net.eval().to(self.device)
        image_gradient_criterion.device = self.device
        torch.quantization.convert(self.net, inplace=True)
        
        print('After quantize')
        self.device = torch.device('cpu')
        iou, loss = self.val(image_gradient_criterion, bce_criterion)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.jit.save(torch.jit.script(self.net), f'{self.checkpoint_dir}/quantized.pt')

        return self.net

                
    def _train_one_epoch(self, epoch, image_gradient_criterion, bce_criterion, quantize=False):
        bce_losses = AverageMeter()
        image_gradient_losses = AverageMeter()
        iou_avg = AverageMeter()
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
            
            iou = iou_loss(pred, mask)

            bce_losses.update(bce_loss.item(), self.batch_size)
            image_gradient_losses.update(self.gradient_loss_weight * image_gradient_loss, self.batch_size)
            iou_avg.update(iou)
            # save sample images
            pbar.set_description(f"Epoch: [{epoch}/{self.num_epoch}] | Bce Loss: {bce_losses.avg:.4f} | "
                f"Image Gradient Loss: {image_gradient_losses.avg:.4f} | IOU: {iou_avg.avg:.4f}")
            
            if step % self.sample_step == 0:
                self.save_sample_imgs(image[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, step)
                # print('[*] Saved sample images')
        if not quantize:
            save_info = {'model': self.net, 'state_dict': self.net.state_dict(), 'optimizer' : self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(save_info, f'{self.checkpoint_dir}/last.pt')

        return iou_avg.avg, bce_losses.avg + image_gradient_losses.avg * self.gradient_loss_weight


    def val(self, image_gradient_criterion, bce_criterion):
        self.net = self.net.eval()
        torch.save(self.net.state_dict(), "tmp.pt")
        model_size = "%.2f MB" %(os.path.getsize("tmp.pt") / 1e6)
        with torch.no_grad():
            bce_losses = AverageMeter()
            image_gradient_losses = AverageMeter()
            inference_avg = AverageMeter()
            iou_avg = AverageMeter()
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            
            for step, (image, gray_image, mask) in pbar:
                image = image.to(self.device)
                mask = mask.to(self.device)
                gray_image = gray_image.to(self.device)

                cur = time.time()
                pred = self.net(image)
                inference_avg.update(time.time() - cur)

                pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()
    
                # preds_flat.shape (N*224*224, 2)
                # masks_flat.shape (N*224*224, 1)
                image_gradient_loss = image_gradient_criterion(pred, gray_image)
                bce_loss = bce_criterion(pred_flat, mask_flat)
    
                loss = bce_loss + self.gradient_loss_weight * image_gradient_loss
                iou = iou_loss(pred, mask)
                
                bce_losses.update(bce_loss.item(), self.batch_size)
                image_gradient_losses.update(self.gradient_loss_weight * image_gradient_loss, self.batch_size)
                iou_avg.update(iou)
    
                pbar.set_description(f"Validate... Bce Loss: {bce_losses.avg:.4f} | "
                        f"Image Gradient Loss: {image_gradient_losses.avg:.4f} | IOU: {iou:.4f} | "
                        f"Model Size: {model_size} | Infernece Speed: {inference_avg.avg:.4f}")
        
        os.remove("tmp.pt")
        self.net = self.net.train()
        return iou_avg.avg, bce_losses.avg + image_gradient_losses.avg * self.gradient_loss_weight
        
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
