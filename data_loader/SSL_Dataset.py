import torch
import numpy as np
from PIL import Image

class SSL_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None, mean=None, std=None):
        self.transform =transform
        self.target_transform = target_transform
        self.mean = mean
        self.std  = std

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def _transpose(self, x, source='NHWC', target='NCHW'):
        return x.transpose([source.index(d) for d in target]) 

    def _get_PIL(self, x):
        return Image.fromarray(x)
        
    def _normalize(self, x):
        x, mean, std = [np.array(a, np.float32) for a in (x, self.mean, self.std)]
        x -= mean*255
        x *= 1.0/(255*std)
        return x
