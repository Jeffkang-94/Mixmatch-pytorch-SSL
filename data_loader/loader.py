import numpy as np
import torchvision
from data_loader.cifar import CIFAR_labeled, CIFAR_unlabeled

class K_Augmentation:
    def __init__(self, K, transform):
        self.transform = transform
        self.K = K

    def __call__(self, x):
        out = [self.transform(x) for _ in range(self.K)]
        return out

def get_trainval_data(root, dataset, K, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):
    if dataset=='CIFAR10':
        base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    elif dataset=='CIFAR100':
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
    elif dataset =='SVHN':
        pass
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))
    train_labeled_dataset = CIFAR_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR_unlabeled(root, train_unlabeled_idxs, train=True, transform=K_Augmentation(K, transform_train))
    val_dataset = CIFAR_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

def get_test_data(root, transform=None, download=True):
    test_dataset = CIFAR_labeled(root, train=False, transform=transform, download=True)
    return test_dataset
    

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs
