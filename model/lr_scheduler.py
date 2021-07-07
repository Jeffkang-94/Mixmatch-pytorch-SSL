#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np


class WarmupCosineLrScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


if __name__ == "__main__":
    '''
     For testing
    '''
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 20000
    lr_scheduler = WarmupCosineLrScheduler(optim, max_iter, 0)

    lrs = []
    for _ in range(max_iter):
        lr = lr_scheduler.get_lr()[0]
        lrs.append(lr)
        lr_scheduler.step()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    lrs = np.array(lrs)
    n_lrs = len(lrs)
    plt.plot(np.arange(n_lrs), lrs)
    plt.title('CosineLrScheduler')
    plt.grid()
    plt.show()
