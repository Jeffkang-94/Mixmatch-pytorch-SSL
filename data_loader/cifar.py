import numpy as np
from data_loader.SSL_Dataset import SSL_Dataset


class CIFAR_labeled(SSL_Dataset):
    def __init__(self, data, targets, indexes=None, transform=None, target_transform=None, mean=None, std=None):
        super(CIFAR_labeled, self).__init__(transform=transform, target_transform=target_transform, mean=mean, std=std)
        self.data = data
        self.targets = targets
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]
        self.data = self._transpose(self._normalize(self.data))

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
    def __init__(self, data, targets, indexes=None,
                 transform=None, target_transform=None, mean=None, std=None):
        super(CIFAR_unlabeled, self).__init__(data, targets, indexes, transform=transform, target_transform=target_transform, mean=mean, std=std)
        self.targets = np.array([-1 for _ in range(len(self.targets))])

