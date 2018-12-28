#
#
# data folder 구조
# (data_folder) / original
# (data_folder) / mask
# (data_folder) / ...
# (data_folder) / ...
#
#
import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception(" ! %s  not exists." % self.data_folder)

        self.objects_path = []
        self.image_name = os.listdir(os.path.join(data_folder, "images"))
        if len(self.image_name) == 0:
            raise Exception("No image found in %s" % self.image_name)
        for p in os.listdir(data_folder):
            if p == "images":
                continue
            self.objects_path.append(os.path.join(data_folder, p))


        self.image_size = image_size

    def activator_binmasks(images, image, augmenter, parents, default):
        if augmenter.name in ["Multiply", "GaussianBlur", "CoarseDropout"]:
            return False
        else:
            return default

    def image_aug(self, image, mask):
        hooks_binmasks = ia.HooksImages(activator=self.activator_binmasks)
        seq = iaa.SomeOf((1, 2),
                         [
                             iaa.OneOf([iaa.Affine(rotate=(-30, 30), name="Rotate"),
                                        iaa.Affine(scale=(0.3, 1.3), name="Scale")]),
                             iaa.OneOf([iaa.Multiply((0.5, 1.5), name="Multiply"),
                                        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
                                        iaa.CoarseDropout((0.05, 0.2), size_percent=(0.01, 0.1),
                                                          name="CoarseDropout")])
                         ])

        seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
        image_aug = np.squeeze(seq_det.augment_images(np.expand_dims(np.array(image), axis=0)), axis=0)
        mask_aug = np.squeeze(seq_det.augment_images((np.expand_dims(np.array(mask), axis=0)), hooks=hooks_binmasks), axis=0)

        return Image.fromarray(image_aug), Image.fromarray(mask_aug)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_folder, 'images', self.image_name[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_folder, 'masks', self.image_name[index]))

        image, mask = self.image_aug(image, mask)


        transform_image = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image)

        transform_mask = (transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])(mask))


        return transform_image, transform_mask #for hair segmentation

    def __len__(self):
        return len( self.image_name)


def get_loader(data_folder, batch_size, image_size, shuffle, num_workers):
    dataset = Dataset(data_folder, image_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)
    return dataloader
