import torch
from model.transfer_model import MobileHairNet


model = MobileHairNet()

model.eval()
image = torch.rand(1, 3, 224, 224)
jit_model = torch.jit.trace(model, image)


jit_model.save("segmentation.pt")

