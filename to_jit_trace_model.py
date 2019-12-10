import torch
import torch.nn as nn
from model.transfer_model import MobileHairNet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = MobileHairNet()

    def forward(self, img):
        x = self.base_model(img)
        x = x.argmax(dim=1)
        x = x.unsqueeze(dim=0)
        x = x.repeat_interleave(3, dim=1)
        x = x * img
        return x


model = Model()
model.eval()
image = torch.rand(1, 3, 224, 224)
jit_model = torch.jit.trace(model, image)


jit_model.save("segmentation.pt")

