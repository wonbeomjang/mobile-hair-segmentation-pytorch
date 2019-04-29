from model import MobileHairNet
from loss import HairMatLoss
import os
from glob import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from loss import iou_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        self.build_model()
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir

    def build_model(self):
        self.net = MobileHairNet()
        self.net.apply(weights_init)
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print(" * Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print(" ! No checkpoint in ", str(self.model_path))
            return

        model = glob(os.path.join(self.model_path, "MobileHairNet*.pth"))
        model.sort()

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(" * Load Model from %s: " % str(self.model_path), str(model[-1]))

    def train(self):
        MobileHairNetLoss = HairMatLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-7)

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

                print("epoch: [%d/%d] | image: [%d/%d] | loss: %.4f | IOU: %.4f" % (epoch, self.epoch, step, self.image_len, loss, iou))

                # save sample images
                if step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, step)
                    print('[*] Saved sample images')

            torch.save(self.net.state_dict(), '%s/MobileHairNet_epoch-%d.pth' % (self.checkpoint_dir, epoch))

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
