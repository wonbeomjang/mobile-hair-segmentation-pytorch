import cv2
import argparse

import torchvision.transforms.functional as TF
import numpy as np

from models import *


def get_mask(image, net, size=224):
    image_h, image_w = image.shape[0], image.shape[1]

    down_size_image = cv2.resize(image, (size, size))
    down_size_image = cv2.cvtColor(down_size_image, cv2.COLOR_BGR2RGB)
    down_size_image = torch.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    down_size_image = np.transpose(down_size_image, (0, 3, 1, 2)).to(device)
    down_size_image = TF.normalize(down_size_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    mask: torch.nn.Module = net(down_size_image)

    # mask = torch.squeeze(mask[:, 1, :, :])
    mask = mask.argmax(dim=1).squeeze()
    mask_cv2 = mask.data.cpu().numpy() * 255
    mask_cv2 = mask_cv2.astype(np.uint8)
    mask_cv2 = cv2.resize(mask_cv2, (image_w, image_h))

    return mask_cv2


def build_model(model_version, quantize, model_path, device) -> nn.Module:
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


def load_model(net, model_path, device) -> nn.Module:
    if model_path:
        print(f'[*] Load Model from {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])

    return net
    

def alpha_image(image, mask, alpha=0.1):
    color = np.zeros((mask.shape[0], mask.shape[1], 3))
    color[np.where(mask != 0)] = [0, 130, 255]
    alpha_hand = ((1 - alpha) * image + alpha * color).astype(np.uint8)
    alpha_hand = cv2.bitwise_and(alpha_hand, alpha_hand, mask=mask)

    return cv2.add(alpha_hand, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=int, default=2, help='MobileHairNet version')
    parser.add_argument('--quantize', nargs='?', const=True, default=False, help='load and train quantizable model')
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.quantize else "cpu")
    net = build_model(args.model_version, args.quantize, args.model_path, device)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise Exception("webcam is not detected")

    while True:
        # ret : frame capture결과(boolean)
        # frame : Capture한 frame

        ret, image = cam.read()

        if ret:
            mask = get_mask(image, net)
            add = alpha_image(image, mask)
            cv2.imshow('frame', add)
            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break

    cam.release()
    cv2.destroyAllWindows()
