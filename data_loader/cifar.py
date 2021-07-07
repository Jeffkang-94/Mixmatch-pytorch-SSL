import logging

import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from data_loader import transform as T

from data_loader.randaugment import RandomAugment
from torch.utils.data import RandomSampler, BatchSampler
#from data_loader.sampler import RandomSampler, BatchSampler

logger = logging.getLogger(__name__)


def load_data_train(L=250, num_val=5000, dataset='CIFAR10', dspth='./data'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [40, 250, 4000], "The number of labels should be 40, 250, or 4000"
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
        assert L in [400, 2500, 10000], "The number of labels should be 400, 2500, or 10000"

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
    n_vals = num_val // n_class
    data_x, label_x, data_u, label_u, data_val, label_val = [], [], [], [], [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        p = len(indices) - n_vals
        inds_x, inds_u, inds_val = indices[:n_labels], indices[n_labels:p], indices[p:]
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
        data_val += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
            for i in inds_val
        ]
        label_val += [labels[i] for i in inds_val]
    return data_x, label_x, data_u, label_u, data_val, label_val


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
    return data, labels


def compute_mean_var(L=40, dataset='CIFAR10'):
    data_x, label_x, data_u, label_u, data_val, label_val = load_data_train(L=L, dataset=dataset)
    data = data_x + data_u + data_val
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, is_train=True):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.is_train = is_train
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.is_train:
            return self.trans_weak(im), self.trans_strong(im), lb
        else:
            return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


class CifarMixMatch(Dataset):
    def __init__(self, dataset, data, labels, is_train=True):
        super(CifarMixMatch, self).__init__()
        self.data, self.labels = data, labels
        self.is_train = is_train
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if is_train:
            self.trans_train = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.is_train:
            return self.trans_train(im), self.trans_train(im), lb
        else:
            return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, L, num_val, root='data'):
    data_x, label_x, data_u, label_u, data_val, label_val = load_data_train(L=L, num_val=num_val, dataset=dataset, dspth=root)

    ds_x = Cifar(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        is_train=True
    )
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )

    ds_u = Cifar(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        is_train=True
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)  # multiply by mu
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )

    ds_val = Cifar(
        dataset=dataset,
        data=data_val,
        labels=label_val,
        is_train=False
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        shuffle=False,
        batch_size=100,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info('labeled sample: {}. unlabeled sample: {}.'.format(len(sampler_x), len(sampler_u)))
    logger.info('labeled loader: {}. unlabeled loader: {}. validation loader: {}'.format(len(dl_x), len(dl_u), len(dl_val)))

    return dl_x, dl_u, dl_val


def get_test_loader(dataset, batch_size, num_workers, pin_memory=True):
    data, labels = load_data_test(dataset)
    logger.info('test dataset: {}.'.format(len(data)))
    ds = Cifar(
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
    logger.info('test loader: {}.'.format(len(dl)))
    return dl


def get_train_loader_mixmatch(dataset, batch_size, mu, n_iters_per_epoch, L, num_val, root='data'):
    data_x, label_x, data_u, label_u, data_val, label_val = load_data_train(L=L, num_val=num_val, dataset=dataset, dspth=root)

    ds_x = CifarMixMatch(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        is_train=True
    )
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )

    ds_u = CifarMixMatch(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        is_train=True
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)  # multiply by mu
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )

    ds_val = CifarMixMatch(
        dataset=dataset,
        data=data_val,
        labels=label_val,
        is_train=False
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        shuffle=False,
        batch_size=100,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info('labeled sample: {}. unlabeled sample: {}.'.format(len(sampler_x), len(sampler_u)))
    logger.info('labeled loader: {}. unlabeled loader: {}. validation loader: {}'.format(len(dl_x), len(dl_u), len(dl_val)))

    return dl_x, dl_u, dl_val


def get_test_loader_mixmatch(dataset, batch_size, num_workers, pin_memory=True):
    data, labels = load_data_test(dataset)
    logger.info('test dataset: {}.'.format(len(data)))
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
    logger.info('test loader: {}.'.format(len(dl)))
    return dl
