
import os.path as osp
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from data_loader import transform as T

from data_loader.randaugment import RandomAugment
from torch.utils.data import RandomSampler, BatchSampler

def load_data_train(L=250, dataset='CIFAR10', dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // n_class
    data_x, label_x, data_u, label_u, data_val, label_val = [], [], [], [], [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]

    print("label set : {}, unlabel set : {}".format(len(label_x), len(label_u)))
    return {'x': data_x, 'y': label_x, 'u_x': data_u, 'u_y': label_u}# 'v_x': data_val, 'v_y': label_val}


def load_data_test(dataset, dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)    # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
        for el in data
    ]
    return {'x': data, 'y': labels}



class CifarMixMatch(Dataset):
    def __init__(self, K, L, dataset, datapath, is_train=True):
        super(CifarMixMatch, self).__init__()
        self.isTrain = is_train
        self.K = K
        
        if self.isTrain:
            self.data = load_data_train(L=L, dataset=dataset, dspth=datapath)
        else:
            self.data = load_data_test(dataset=dataset, dspth=datapath)

        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if self.isTrain:
            self.trans_train = T.Compose([
                #T.Resize((32, 32)),
                T.PadandRandomCrop(cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                #T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        if self.isTrain:
            u_idx = random.randint(0, len(self.data['u_x'])-1)
            x   ,  y = self.data['x'][idx], self.data['y'][idx]
            u_x, u_y = self.data['u_x'][u_idx], self.data['u_y'][u_idx]
            x = self.trans_train(x)
            u_x_ = []
            for _ in range(self.K):
                u_x_.append(self.trans_train(self.data['u_x'][u_idx]))
            u_y = [u_y] *self.K
            return x, u_x_, y, u_y
        else:
            img, y = self.trans(self.data['x'][idx]) , self.data['y'][idx]
            return img, y

    def __len__(self):
        leng = len(self.data['x'])
        return leng


def get_train_loader_mixmatch(dataset, K, batch_size, n_iters_per_epoch, L, root='data'):
    ds_train = CifarMixMatch(
        K = K,
        L = L, 
        dataset=dataset,
        datapath =root,
        is_train=True
    )
    sampler_x = RandomSampler(ds_train, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )
    ds_test = CifarMixMatch(
        K = K,
        L = L, 
        dataset=dataset,
        datapath =root,
        is_train=False
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )
    return dl_train, dl_test



def get_test_loader_mixmatch(dataset, batch_size, num_workers, pin_memory=True):
    data, labels = load_data_test(dataset)
    ds = CifarMixMatch(
        dataset=dataset,
        data=data,
        labels=labels,
        is_train=False
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl
