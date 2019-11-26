from model.model import MobileHairNet
from loss.loss import ImageGradientLoss
import os
from glob import glob
import torch
from loss.loss import iou_loss
from utils.util import LambdaLR, AverageMeter
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.adadelta import Adadelta
import torch.nn as nn


class Trainer:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = config.epoch
        self.num_epoch = config.num_epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.eps = config.eps
        self.rho = config.rho
        self.decay = config.decay
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        self.gradient_loss_weight = config.gradient_loss_weight
        self.decay_batch_size = config.decay_batch_size

        self.build_model()
        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho, weight_decay=self.decay)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(self.optimizer, LambdaLR(self.num_epoch,
                                                                                                     self.epoch,
                                                                                                     len(self.data_loader),
                                                                                                     self.decay_batch_size).step)

    def build_model(self):
        self.net = MobileHairNet().to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print("[!] No checkpoint in ", str(self.model_path))
            return

        model_path = os.path.join(self.model_path, f"MobileHairNet_epoch-{self.epoch-1}.pth")
        model = glob(model_path)
        model.sort()
        if not model:
            print(f"[!] No Checkpoint in {model_path}")
            return

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(f"[*] Load Model from {model[-1]}: ")

    def train(self):
        bce_losses = AverageMeter()
        image_gradient_losses = AverageMeter()
        image_gradient_criterion = ImageGradientLoss().to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epoch, self.num_epoch):
            bce_losses.reset()
            image_gradient_losses.reset()
            for step, (image, gray_image, mask) in enumerate(self.data_loader):
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
                if step % 10 == 0:
                    print(f"Epoch: [{epoch}/{self.num_epoch}] | Step: [{step}/{self.image_len}] | "
                          f"Bce Loss: {bce_losses.avg:.4f} | Image Gradient Loss: {image_gradient_losses.avg:.4f} | "
                          f"IOU: {iou:.4f}")
                if step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, step)
                    print('[*] Saved sample images')

            torch.save(self.net.state_dict(), f'{self.checkpoint_dir}/MobileHairNet_epoch-{epoch}.pth')

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