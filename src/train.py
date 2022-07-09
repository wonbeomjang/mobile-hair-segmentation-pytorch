import os
import time
import traceback

import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.adadelta import Adadelta
from torch.optim.lr_scheduler import OneCycleLR

from models import *
from models.quantization.modelv1 import QuantizableMobileHairNet
from models.quantization.modelv2 import QuantizableMobileHairNetV2
from loss.loss import ImageGradientLoss, iou_loss
from utils.util import AverageMeter, get_model_size

from data.dataloader import get_loader


class Trainer:
    def __init__(self, config):
        self.net = None
        self.run = None
        self.lr_scheduler = None
        self.optimizer = None
        self.data_loader, self.val_loader = get_loader(config.data_path, config.batch_size, config.image_size,
                                                       shuffle=True, num_workers=int(config.workers),
                                                       seed=config.manual_seed)
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = 0
        self.num_epoch = config.num_epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_len = len(self.data_loader)
        self.num_classes = config.num_classes
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        self.gradient_loss_weight = config.gradient_loss_weight
        self.model_version = config.model_version
        self.quantize = config.quantize
        self.resume = config.resume
        self.eps = config.eps
        self.rho = config.rho
        self.decay = config.decay
        self.num_quantize_train = config.num_quantize_train
        self.loss = float('inf')

        self.build_model()

    def build_model(self):
        if self.model_version == 1:
            if self.quantize:
                self.net = quantized_modelv1().to(self.device)
            else:
                self.net = modelv1().to(self.device)
        elif self.model_version == 2:
            if self.quantize:
                self.net = quantized_modelv2().to(self.device)
            else:
                self.net = modelv2().to(self.device)

        else:
            raise Exception('[!] Unexpected model version')

        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho,
                                  weight_decay=self.decay)
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.lr, epochs=self.num_epoch,
                                       steps_per_epoch=self.image_len, cycle_momentum=False)
        self.load_model()

    def load_model(self):
        if not self.model_path and not self.resume:
            self.run = wandb.init(project='hair_segmentation', dir=os.getcwd())
            return

        ckpt = f'{self.checkpoint_dir}/last.pth' if self.resume else self.model_path
        save_info: dict = torch.load(ckpt, map_location=self.device)
        run_id = save_info['run_id'] if 'run_id' in save_info else None
        self.run = wandb.init(id=run_id, project='hair_segmentation', resume="allow", dir=os.getcwd())

        # try:
        #     save_info = torch.load(wandb.restore(ckpt).name, map_location=self.device)
        # except ValueError:
        #     print(traceback.format_exc())
        #     print(f"[!] {ckpt} is not exist in wandb")
        self.epoch = save_info['epoch'] + 1
        # self.net = save_info['model']

        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho,
                                  weight_decay=self.decay)
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.lr, epochs=self.num_epoch,
                                       steps_per_epoch=self.image_len, cycle_momentum=False)

        self.optimizer.load_state_dict(save_info['optimizer'])
        self.net.load_state_dict(save_info['state_dict'])
        self.lr_scheduler.load_state_dict(save_info['lr_scheduler'])
        self.loss = save_info['loss']

        print(f"[*] Load Model from {ckpt}")

    def train(self):
        image_gradient_criterion = ImageGradientLoss().to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)
        best = self.loss

        for epoch in range(self.epoch, self.num_epoch):
            results = self._train_one_epoch(epoch, image_gradient_criterion, bce_criterion)

            save_info = {'model': self.net, 'state_dict': self.net.state_dict(),
                         'optimizer': self.optimizer.state_dict(), 'epoch': epoch,
                         'lr_scheduler': self.lr_scheduler.state_dict(), 'run_id': self.run.id,
                         'loss': results["train/loss"]}

            if self.val_loader:
                val_results = self.val(image_gradient_criterion, bce_criterion)
                results.update(val_results)
                save_info['loss'] = val_results["val/loss"]

                if save_info['loss'] < best:
                    best = save_info['loss']
                    torch.save(save_info, f'{self.checkpoint_dir}/best.pth')
                    wandb.save(f'{self.checkpoint_dir}/best.pth', './', 'now')
                    torch.jit.script(self.net).save(f'{self.checkpoint_dir}/best.pt')

            torch.save(save_info, f'{self.checkpoint_dir}/last.pth')
            wandb.save(f'{self.checkpoint_dir}/last.pth', './')
            wandb.log(results)

        print(f'Final Loss: {best:.4f}')
        self.run.finish()

    def quantize_model(self):
        if not self.quantize:
            return

        ckpt = f'{self.checkpoint_dir}/best.pth'
        print(f'[*] Load Best Model in {ckpt}')
        save_info = torch.load(ckpt, map_location=self.device)
        # self.net = save_info['model']
        self.net.load_state_dict(save_info['state_dict'])

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
        self.net.eval()
        self.net.fuse_model()
        self.net.train()
        self.net = torch.quantization.prepare_qat(self.net)

        temp = self.num_epoch
        self.num_epoch = self.num_quantize_train
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.lr, epochs=self.num_epoch,
                                       steps_per_epoch=self.image_len, cycle_momentum=False)

        for i in range(self.num_quantize_train):
            self._train_one_epoch(i, image_gradient_criterion, bce_criterion, quantize=True)
        self.num_epoch = temp

        self.device = torch.device('cpu')
        self.net = self.net.eval().to(self.device)
        image_gradient_criterion.device = self.device
        self.net = torch.quantization.convert(self.net)

        print('After quantize')
        self.device = torch.device('cpu')
        self.val(image_gradient_criterion, bce_criterion)

        save_info = {'model': self.net, 'state_dict': self.net.state_dict()}
        torch.save(save_info, f'{self.checkpoint_dir}/quantized.pth')

        net = torch.jit.script(self.net)
        print("[*] Save quantized model")
        torch.jit.save(net, f'{self.checkpoint_dir}/quantized.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return net

    def _train_one_epoch(self, epoch, image_gradient_criterion, bce_criterion, quantize=False):
        bce_losses = AverageMeter()
        image_gradient_losses = AverageMeter()
        iou_avg = AverageMeter()
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        image, mask, pred = None, None, None
        results = {}

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
            self.lr_scheduler.step()

            iou = iou_loss(pred, mask)
            bce_losses.update(bce_loss.item(), self.batch_size)
            image_gradient_losses.update(self.gradient_loss_weight * image_gradient_loss, self.batch_size)
            iou_avg.update(iou)
            # save sample images
            pbar.set_description(f"Epoch: [{epoch}/{self.num_epoch}] | Bce Loss: {bce_losses.avg:.4f} | "
                                 f"Image Gradient Loss: {image_gradient_losses.avg:.4f} | IOU: {iou_avg.avg:.4f}")

        if image is not None:
            img = torch.cat(
                [image[0], mask[0].repeat(3, 1, 1), pred[0].argmax(dim=0).unsqueeze(dim=0).repeat(3, 1, 1)], dim=2)
            results["prediction"] = wandb.Image(img)
        results["train/iou"] = iou_avg.avg
        results["train/loss"] = bce_losses.avg + image_gradient_losses.avg * self.gradient_loss_weight

        return results

    def val(self, image_gradient_criterion, bce_criterion):
        self.net = self.net.eval()

        model_size = get_model_size(self.net)
        results = {}
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
                                     f"Model Size: {model_size:.2f}MB | Infernece Speed: {inference_avg.avg:.4f}")

        self.net = self.net.train()
        results["val/iou"] = iou_avg.avg
        results["val/loss"] = bce_losses.avg + image_gradient_losses.avg * self.gradient_loss_weight

        return results
