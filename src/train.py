from model.model import MobileHairNet
from loss.loss import HairMatLoss
import os
from glob import glob
import torch
from loss.loss import iou_loss
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.adadelta import Adadelta


class Trainer:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = config.epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.num_classes = config.num_classes
        self.eps = config.eps
        self.rho = config.rho
        self.decay = config.decay
        self.build_model()
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir

    def build_model(self):
        self.net = MobileHairNet()
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print("[!] No checkpoint in ", str(self.model_path))
            return

        model = glob(os.path.join(self.model_path, "MobileHairNet*.pth"))
        model.sort()

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print("[*] Load Model from %s: " % str(self.model_path), str(model[-1]))

    def train(self):
        MobileHairNetLoss = HairMatLoss().to(self.device)
        optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho, weight_decay=self.decay)

        for epoch in range(self.epoch):
            for step, (image, mask) in enumerate(self.data_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)
                pred = self.net(image)

                self.net.zero_grad()
                loss = MobileHairNetLoss(pred, image, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iou = iou_loss(pred, mask)

                print(f"epoch: [{epoch}/{self.epoch}] | image: [{step}/{self.image_len}] | loss: {loss:.4f} | "
                      f"IOU: {iou:.4f}" )

                # save sample images
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