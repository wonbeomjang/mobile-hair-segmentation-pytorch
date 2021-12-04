import torch

from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F


class ImageGradientLoss(_WeightedLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', device=None):
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, gray_image):
        size = pred.size()
        pred = pred.argmax(1).view(size[0], 1, size[2], size[3]).float()
        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x)
        G_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_image, gradient_tensor_y)
        G_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / torch.sum(G)

        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0

        return image_gradient_loss


def iou_loss(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return loss
