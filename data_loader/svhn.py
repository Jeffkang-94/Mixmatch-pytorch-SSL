import numpy as np
from data_loader.SSL_Dataset import SSL_Dataset
from PIL import Image

class SVHN_labeled(SSL_Dataset):
    def __init__(self, data, targets, indexes=None, transform=None, target_transform=None):
        super(SVHN_labeled, self).__init__(transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = targets
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1,2,0)))
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                img = self.transform[0](img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        leng = len(self.data)
        return leng

class SVHN_unlabeled(SVHN_labeled):
    def __init__(self, data, targets, indexes=None,
                 transform=None, target_transform=None):
        super(SVHN_unlabeled, self).__init__(data, targets, indexes, transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for _ in range(len(self.targets))])

