#
#
# data folder 구조
# (data_folder) / images
# (data_folder) / masks
# (data_folder) / ...
# (data_folder) / ...
#
#
import os

from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
from data.utils import check_data


def transform(image, mask, image_size=224):
    # # Resize
    # resized_num = int(random.random() * image_size)
    # image = TF.resize(image, [image_size + resized_num, image_size + resized_num])
    # mask = TF.resize(mask, [image_size + resized_num, image_size + resized_num])
    #
    # num_pad = int(random.random() * image_size)
    # image = TF.pad(image, num_pad, padding_mode='edge')
    # mask = TF.pad(mask, num_pad)

    # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=(image_size, image_size))
    # image = TF.crop(image, i, j, h, w)
    # mask = TF.crop(mask, i, j, h, w)
    #
    # # Random horizontal flipping
    # if random.random() > 0.5:
    #     image = TF.hflip(image)
    #     mask = TF.hflip(mask)
    #
    # degree = random.random() * 360
    #
    # image = TF.rotate(image, degree)
    # mask = TF.rotate(mask, degree)
    #
    # # Random vertical flipping
    # if random.random() > 0.5:
    #     image = TF.vflip(image)
    #     mask = TF.vflip(mask)

    resize = transforms.Resize(size=(image_size, image_size))
    image = resize(image)
    mask = resize(mask)

    # Make gray scale image
    gray_image = TF.to_grayscale(image)
    mask = TF.to_grayscale(mask)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    gray_image = TF.to_tensor(gray_image)

    # Normalize Data
    image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return image, gray_image, mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception(f"[!] {self.data_folder} not exists.")

        self.objects_path = []
        self.image_name = check_data(data_folder)
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

        image, gray_image, mask = transform(image, mask)

        return image, gray_image, mask

    def __len__(self):
        return len(self.image_name)


def get_loader(data_folder, batch_size, image_size, shuffle, num_workers, seed=100):
    dataset = Dataset(data_folder, image_size)

    dataset, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.95),
                                                               len(dataset) - int(len(dataset) * 0.95)
                                                               ],
                                                     generator=torch.Generator().manual_seed(seed))

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return data_loader, val_loader
