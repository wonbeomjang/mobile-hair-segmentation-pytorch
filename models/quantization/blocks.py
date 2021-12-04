import torch.nn as nn
from torch.quantization import fuse_modules

from ..blocks import 

class QuantizableLayerDepwiseEncode(LayerDepwiseEncode):
    def __init__(self, in_channels, out_channels, kernel_size=3, reserve=False):
        super(QuantizableLayerDepwiseEncode, self).__init__()
        
    def fuse_model(self):
        fuse_modules(self.layer, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer, ["3", "4", "5"], inplace=True)
        

class QuantizableLayerDepwiseDecode(LayerDepwiseDecode):
    def __init__(self, in_channels, out_channels, kernel_size=3, reserve=False):
        super(QuantizableLayerDepwiseEncode, self).__init__()
    
    def fuse_model(self):
        fuse_modules(self.layer, [str(1), str(2)], inplace=True)