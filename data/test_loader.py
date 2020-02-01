import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from random import random


def transform(image, mask, image_size=224):
    resize = transforms.Resize(size=(image_size, image_size))
    image = resize(image)
    mask = resize(mask)

    if random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    if random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    angle = random() * 12 - 6
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)

    pad_size = random() * image_size
    image = TF.pad(image, pad_size, padding_mode='edge')
    mask = TF.pad(mask, pad_size, padding_mode='edge')

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    # Normalize Data
    image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return image, mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception(f"[!] {self.data_folder} not exists.")

        self.objects_path = []
        self.image_name = os.listdir(os.path.join(data_folder, "images"))
        if len(self.image_name) == 0:
            raise Exception(f"No image found in {self.image_name}")
        for p in os.listdir(data_folder):
            if p == "images":
                continue
            self.objects_path.append(os.path.join(data_folder, p))

        self.image_size = image_size

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_folder, 'images', self.image_name[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_folder, 'masks', self.image_name[index]))

        image, mask = transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.image_name)


def get_loader(data_folder, batch_size, image_size, shuffle, num_workers):
    dataset = Dataset(data_folder, image_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader
