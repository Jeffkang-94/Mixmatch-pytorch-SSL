import torch
import numpy as np
from PIL import Image

class SSL_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform =transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def _transpose(self, x, source='NHWC', target='NCHW'):
        return x.transpose([source.index(d) for d in target]) 

    def _get_PIL(self, x):
        return Image.fromarray(x)
        
