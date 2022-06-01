import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from models import *


def build_model(model_version, quantize, model_path, device) :
    if model_version == 1:
        if quantize:
            net = quantized_modelv1(pretrained=True, device=device).to(device)
        else:
            net = modelv1(pretrained=True, device=device).to(device)
    elif model_version == 2:
        if quantize:
            net = quantized_modelv2(pretrained=True, device=device).to(device)
        else:
            net = modelv2(pretrained=True, device=device).to(device)
    else:
        raise Exception('[!] Unexpected model version')

    net = load_model(net, model_path, device)
    return net


def load_model(net, model_path, device):
    if model_path:
        print(f'[*] Load Model from {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])

    return net


def predict(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.quantize else "cpu")
    net = build_model(args.model_version, args.quantize, args.model_path, device).to(device)

    image = Image.open(args.image_path)
    image = TF.to_tensor(image).to(device)
    _, w, h = image.shape
    image = TF.resize(image, [224, 224])
    image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    mask = net(image.unsqueeze(0))
    mask = mask.argmax(dim=1)
    mask = TF.resize(mask, [w, h]).squeeze()

    save_image(mask.float(), args.result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_version', type=int, default=2, help='MobileHairNet version')
    parser.add_argument('--quantize', nargs='?', const=True, default=False, help='load and train quantizable model')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('-i', '--image_path', type=str, default=None, help='path of the image')
    parser.add_argument('-o', '--result_path', type=str, default=None, help='path of the image')

    args = parser.parse_args()

    predict(args)

