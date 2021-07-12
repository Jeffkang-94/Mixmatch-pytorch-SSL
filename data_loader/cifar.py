import torchvision
from data_loader.loader import *
import numpy as np

def normalize(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 



class SSL_Dataset(torchvision.datasets):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        self.root = root
        self.transform =transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class CIFAR_labeled(torchvision.datasets.CIFAR10):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        
            # mean = (0.5071, 0.4867, 0.4408)
            # std  = (0.2675, 0.2565, 0.2761)
        super(CIFAR_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data, mean, std))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        leng = len(self.data)
        return leng

class CIFAR_unlabeled(CIFAR_labeled):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for _ in range(len(self.targets))])

