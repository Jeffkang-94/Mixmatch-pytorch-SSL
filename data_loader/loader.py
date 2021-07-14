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

def train_val_split(labels, n_labeled, num_class, num_val):
    n_labeled_per_class = int(n_labeled/num_class)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-num_val])
        val_idxs.extend(idxs[-num_val:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs
def get_trainval_data(root, dataset, K, n_labeled, num_class,
                 transform_train=None, transform_val=None,
                 download=True):
    if dataset=='CIFAR10':
        """
            TOTAL : 32x32 RGB 60,000 images
                    6,000 images, 10 classes.
                    Training : 50,000 images, 5000 images per each class
                    Test     : 10,000 images, 1000 images per each class
        """
        print("CIFAR10")
        base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
        mean = (0.4914, 0.4822, 0.4465) 
        std  = (0.2471, 0.2435, 0.2616) 
        num_val = 5000 // num_class

        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, n_labeled, num_class, num_val)
        train_labeled_dataset = CIFAR_labeled(base_dataset.data ,base_dataset.targets, train_labeled_idxs,  transform=transform_train, mean=mean, std=std)
        train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=K_Augmentation(K, transform_train), mean=mean, std=std)
        val_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, val_idxs, transform=transform_val, mean=mean, std=std)

    elif dataset=='CIFAR100':
        """
            TOTAL : 32x32 RGB 60,000 images
                    600 images, 100 classes.
                    Training : 50,000 images, 500 images per each class
                    Test     : 10,000 images, 100 images per each class
        """
        print("CIFAR100")
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        num_val = 5000 // num_class

        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, n_labeled, num_class, num_val)
        train_labeled_dataset = CIFAR_labeled(base_dataset.data ,base_dataset.targets, train_labeled_idxs, transform=transform_train, mean=mean, std=std)
        train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=K_Augmentation(K, transform_train), mean=mean, std=std)
        val_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, val_idxs, transform=transform_val, mean=mean, std=std)
        

    elif dataset =='SVHN':
        """
            TOTAL : 32x32 RGB 99,289 + (531,131 extra) images, 10 classes.
                    Training : 73,257 images (531,131 extra)
                    Test     : 26,032 images
        """
        print("SVHN")
        base_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
        mean = (0.4377, 0.4438, 0.4728)
        std  = (0.1980, 0.2010, 0.1970)
        num_val = 5000 // num_class
        raise NotImplementedError
    elif dataset =='STL10':
        """
            TOTAL : 96x96 RGB 113,000 images, 10 classes.
                    Training : 500 images per each class, 10 classes(pre-defined 10 folds). + 100,000 images without label
                    Test     : 800 images per each class, 10 classes
        """
        print("STL10")
        base_dataset = torchvision.datasets.STL10(root, split='train', download=download)
        mean = (0.4409, 0.4279, 0.3868)   
        std  = (0.2683, 0.2611, 0.2687)
        num_val = 5000 // num_class
        raise NotImplementedError
    
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

def get_test_data(root, dataset, transform_val):
    if dataset=='CIFAR10':
        base_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True)
        mean = (0.4914, 0.4822, 0.4465) 
        std  = (0.2471, 0.2435, 0.2616)
        test_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, None, transform=transform_val, mean=mean, std=std)
    elif dataset=='CIFAR100':
        base_dataset = torchvision.datasets.CIFAR100(root, train=False, download=True)
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        test_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, None, transform=transform_val, mean=mean, std=std)
    elif dataset=='SVHN':
        base_dataset = torchvision.datasets.SVHN(root, split='test', download=True)
        mean = (0.4377, 0.4438, 0.4728)
        std  = (0.1980, 0.2010, 0.1970)
        raise NotImplementedError
    elif dataset=='STL10':
        base_dataset = torchvision.datasets.STL10(root, split='test', download=True)
        mean = (0.4409, 0.4279, 0.3868)   
        std  = (0.2683, 0.2611, 0.2687)
        raise NotImplementedError
    return test_dataset

