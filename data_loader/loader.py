import numpy as np
import torchvision
from data_loader.cifar import CIFAR_labeled, CIFAR_unlabeled
from data_loader.svhn import SVHN_labeled, SVHN_unlabeled

class Augmentation:
    def __init__(self, K, transform):
        self.transform = transform
        self.K = K

    def __call__(self, x):
        # Applying stochastic augmentation K times
        out = [self.transform(x) for _ in range(self.K)]
        return out

class Fixmatch_Augmentation:
    def __init__(self, transform):
        self.weak_Tranform = transform[0]
        self.strong_Transform = transform[1]

    def __call__(self, x):
        #out = [self.weak_Tranform(x), self.strong_Transform(x)]
        weak_x = self.weak_Tranform(x)
        strong_x = self.strong_Transform(x)
        return (weak_x, strong_x)

def train_val_split(labels, n_labeled, num_class, num_val):
    # depends on the seed, which means the performance 
    # might be changed if you use different seed value.
    n_labeled_per_class = int(n_labeled/num_class)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        if num_val == 0:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
        else:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-num_val])
            val_idxs.extend(idxs[-num_val:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    if not num_val == 0:
        np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def get_trainval_data(root, method, dataset, K, n_labeled, num_class,
                 transform_train=None, transform_val=None,
                 download=True):
    if dataset=='CIFAR10':
        """
            TOTAL : 32x32 RGB 60,000 images
                    6,000 images, 10 classes.
                    Training : 50,000 images, 5000 images per each class
                    Test     : 10,000 images, 1000 images per each class
        """
        print("Dataset : CIFAR10")
        base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
        num_val = 5000 // num_class

        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, n_labeled, num_class, num_val)
        train_labeled_dataset = CIFAR_labeled(base_dataset.data ,base_dataset.targets, train_labeled_idxs,  transform=transform_train)
        if method == 'Mixmatch':
            train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=Augmentation(K,transform_train))
        elif method =='Fixmatch':
            train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=Fixmatch_Augmentation(transform_train))
        val_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, val_idxs, transform=transform_val)

    elif dataset=='CIFAR100':
        """
            TOTAL : 32x32 RGB 60,000 images
                    600 images, 100 classes.
                    Training : 50,000 images, 500 images per each class
                    Test     : 10,000 images, 100 images per each class
        """
        print("Dataset : CIFAR100")
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
        num_val = 5000 // num_class

        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, n_labeled, num_class, num_val)
        train_labeled_dataset = CIFAR_labeled(base_dataset.data ,base_dataset.targets, train_labeled_idxs, transform=transform_train)
        if method == 'Mixmatch':
            train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=Augmentation(K,transform_train))
        elif method =='Fixmatch':
            train_unlabeled_dataset = CIFAR_unlabeled(base_dataset.data , base_dataset.targets, train_unlabeled_idxs, transform=Fixmatch_Augmentation(transform_train))
        val_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, val_idxs, transform=transform_val)
        

    elif dataset =='SVHN':
        """
            TOTAL : 32x32 RGB 99,289 + (531,131 extra) images, 10 classes.
                    Training : 73,257 images (531,131 extra)
                    Test     : 26,032 images
        """
        print("Dataset : SVHN")
        base_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
        
        #num_val = 5000 // num_class
        num_val = 0 
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.labels, n_labeled, num_class, num_val)
        train_labeled_dataset = SVHN_labeled(base_dataset.data ,base_dataset.labels, train_labeled_idxs, transform=transform_train)
        if method == 'Mixmatch':
            train_unlabeled_dataset = SVHN_unlabeled(base_dataset.data , base_dataset.labels, train_unlabeled_idxs, transform=Augmentation(K,transform_train))
        elif method =='Fixmatch':
            train_unlabeled_dataset = SVHN_unlabeled(base_dataset.data , base_dataset.labels, train_unlabeled_idxs, transform=Fixmatch_Augmentation(transform_train))
        #val_dataset = SVHN_labeled(base_dataset.data, base_dataset.labels, val_idxs, transform=transform_val)
        val_dataset = get_test_data(root, dataset, transform_val)
    elif dataset =='STL10':
        """
            TOTAL : 96x96 RGB 113,000 images, 10 classes.
                    Training : 500 images per each class, 10 classes(pre-defined 10 folds). + 100,000 images without label
                    Test     : 800 images per each class, 10 classes
        """
        print("Dataset : STL10")
        base_dataset = torchvision.datasets.STL10(root, split='train', download=download)
        num_val = 5000 // num_class
        raise NotImplementedError
    
    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

def get_test_data(root, dataset, transform_val):
    if dataset=='CIFAR10':
        base_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True)
        test_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, None, transform=transform_val)
    elif dataset=='CIFAR100':
        base_dataset = torchvision.datasets.CIFAR100(root, train=False, download=True)
        test_dataset = CIFAR_labeled(base_dataset.data, base_dataset.targets, None, transform=transform_val)
    elif dataset=='SVHN':
        base_dataset = torchvision.datasets.SVHN(root, split='test', download=True)
        test_dataset = SVHN_labeled(base_dataset.data, base_dataset.labels, None, transform=transform_val)
    elif dataset=='STL10':
        base_dataset = torchvision.datasets.STL10(root, split='test', download=True)
        raise NotImplementedError
    return test_dataset

