from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch
from config.config import get_config
config = get_config()
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

def image_gradient(image):
    edges_x = filters.sobel_h(image)
    edges_y = filters.sobel_v(image)
    edges_x = normalize(edges_x)
    edges_y = normalize(edges_y)
    return torch.from_numpy(edges_x), torch.from_numpy(edges_y)


def image_gradient_loss(image, pred):
    loss = 0
    for i in range(len(image)):
        pred_grad_x, pred_grad_y = image_gradient(pred[i][0].cpu().detach().numpy())
        gray_image = torch.from_numpy(rgb2gray(image[i].permute(1, 2, 0).cpu().numpy()))
        image_grad_x, image_grad_y = image_gradient(gray_image)
        IMx = torch.mul(image_grad_x, pred_grad_x).float()
        IMy = torch.mul(image_grad_y, pred_grad_y).float()
        Mmag = torch.sqrt(torch.add(torch.pow(pred_grad_x, 2), torch.pow(pred_grad_y, 2))).float()
        IM = torch.add(torch.ones(224, 224), torch.neg(torch.pow(torch.add(IMx, IMy), 2)))
        numerator = torch.sum(torch.mul(Mmag, IM))
        denominator = torch.sum(Mmag)
        loss = loss + torch.div(numerator, denominator)
    return torch.div(loss, len(image))


class HairMatLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(HairMatLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss = 0
        self.num_classes = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, image, mask):
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        cross_entropy_loss = F.cross_entropy(pred_flat, mask_flat, weight=self.weight
                                             , ignore_index=self.ignore_index, reduction=self.reduction)
        image_loss = image_gradient_loss(image, pred).to(self.device)
        # return cross_entropy_loss
        return torch.add(cross_entropy_loss, 0.5*image_loss.float())

def iou_loss(pred, mask):
    pred = torch.argmax(pred, 1).long()
    mask = torch.squeeze(mask).long()
    Union = torch.where(pred > mask, pred, mask)
    Overlep = torch.mul(pred, mask)
    loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
    return loss