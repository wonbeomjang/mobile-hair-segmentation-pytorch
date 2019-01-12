# hair-segmentation-pytorch
  <p>This repository is part of a program for previewing your own staining. To do this, you need to separate the hair from the head. So we  borrowed the model structure from the following article.</p>
  
 <p>[A reference article] (https://arxiv.org/abs/1712.07168)</p>
 
## Differences from article
<p>This model is a combination of mobilenet and unet. The difference between this article and the implemented code is that we wrote the loss function without using the function defined in the paper and wrote the cross entropy function and we used Adam as the optimizer. If you want the same code implemented in this article, you can change the commented return value in dyeing-program/loss.py and change the optimizer in dyeing-program/train.py.</p>

## how to retrain
<p>This model can be used for general sementic segmentation with two classes. Create a dataset folder, create a folder with a name other than original in that folder, and run train.py with the original image in orginal, an annotated image in another folder, and so on. At this time, the annotated image enters the two-dimensional image with the black background on the white background.</p>

## Things to update when the time is long
<p>increase the number of classes.</p>
