import numpy as np
import os
import pickle

def load_data(num_label=250, num_val=5000, img_size=32, dataset='CIFAR10', dspth='./data', isTrain=False):
    if isTrain:
        if dataset == 'CIFAR10':
            datalist = [
                os.path.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
                for i in range(5)
            ]
            n_class = 10
            assert L in [40, 250, 4000], "The number of labels should be 40, 250, or 4000"
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
                data[i].reshape(3, img_size, img_size).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
                for i in inds_x
            ]
            label_x += [labels[i] for i in inds_x]
            data_u += [
                data[i].reshape(3, img_size, img_size).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
                for i in inds_u
            ]
            label_u += [labels[i] for i in inds_u]
            data_val += [
                data[i].reshape(3, img_size, img_size).transpose(1, 2, 0)   # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
                for i in inds_val
            ]
            label_val += [labels[i] for i in inds_val]
        return data_x, label_x, data_u, label_u, data_val, label_val
    else:
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
            el.reshape(3, img_size, img_size).transpose(1, 2, 0)    # Transpose to (32, 32, 3) to use cv2 library. T.Tensor will re-transpose it back to (3, 32, 32)
            for el in data
        ]
        return data, labels
    