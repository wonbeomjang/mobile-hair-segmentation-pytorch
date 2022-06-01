# mobile-hair-segmentation-pytorch
This repository is part of a program for previewing your own dyeing on mobile device.
To do this, you need to separate the hair from the head.
And we have to use as light model as MobileNet to use in mobile device in real time.
So we borrowed the model structure from the following article.  
  
[Real-time deep hair matting on mobile devices](https://arxiv.org/abs/1712.07168) 

## model architecture
![network_architecture](./image/network_architecture.PNG)   
This model MobileNet + SegNet.  
To do semantic segmentation they transform MobileNet like SegNet.
And add additional loss function to capture fine hair texture.

## install requirements
```bash
pip install -r requirements.txt
```

## model performance
|                        | IOU (%) | inference speed; batch size:0 (ms) | inference speed; batch size:32 (ms) | model size (MB) |
|:----------------------:|:-------:|:----------------------------------:|:-----------------------------------:|:---------------:|
| version1 (MobilenetV1) |  92.48  |                 10                 |                 2372                |      15.61      |
|  quatization version 1 |  91.51  |                 50                 |                 1682                |       4.40      |
| version2 (MobilenetV2) |  93.21  |                 16                 |                 2461                |      15.27      |
| quantization version 2 |  92.90  |                 37                 |                 754                 |       6.88      |

![network_architecture](./image/sample_image.PNG)
![network_architecture](./image/webcam.gif)

## preparing datsets
make directory like this
```
dataset
   |__ images
   |
   |__ masks
   
```
expected image name  
The name of the expected image pair is:  
```
 - dataset/images/1.jpg 
| 
 - dataset/masks/1.jpg  
```

```
/dataset
    /images
        /1.jpg
        /2.jpg
        /3.jpg 
         ...
    /masks
        /1.jpg
        /2.jpg
        /3.jpg 
         ...
```
## how to train
after 200 epoch, add other commented augmentation and remove resize  
(dataloader/dataloader.py)  

There are modelv1 and modelv2 whose backbone are mobilenetv1 and mobilenetv2 each other.
Default is mobilenetv2
```
python main.py --num_epoch [NUM_EPOCH] --model_version [1~2]
```
If you want to quantize model
```
python main.py --num_epoch [NUM_EPOCH] --model_version [1~2] --quantize
```
Or if you want to resume model training
```
python main.py --num_epoch [NUM_EPOCH] --model_version [1~2] --resume
```
```
python main.py --num_epoch [NUM_EPOCH] --model_version [1~2] --model_path [MODEL_PATH]
```
## Test
```bash
python webcam.py --model_path [MODEL_PATH]
```
if you want to use quantized model
```bash
python webcam.py --model_path [MODEL_PATH] --quantize
```

## Load model
All you have to do is copy ./model and use it

ex)
```python
import torch
from models import quantized_modelv2

quantize = False
device = torch.device("cuda:0" if torch.cuda.is_available() and not quantize else "cpu")
quantized_modelv2(pretrained=True, device=device).to(device)
```

