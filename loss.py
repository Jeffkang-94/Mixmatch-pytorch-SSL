import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np
from utils import *

class MixMatchLoss(nn.Module):
    def __init__(self, alpha, lambda_u, T, K, num_classes, device):
        super().__init__()
        self.K = K
        self.T = T
        self.lambda_u = lambda_u
        self.lamda = Beta(alpha, alpha)
        self.num_classes = num_classes
        self.device = device

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)

    def linear_rampup(self, current):
        rampup_length = 1024
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def cal_loss(self, logit_x, y, logit_u_x, y_hat, epoch, lambda_u):
        probs_u = torch.softmax(logit_u_x, dim=1)
        loss_x = F.cross_entropy(logit_x, y)
        loss_u_x = F.mse_loss(probs_u, y_hat)

        return loss_x, loss_u_x, lambda_u * self.linear_rampup(epoch)  

    def forward(self, input):
        x = input['x']
        y = input['y']
        u_x = [x for x in input['u_x']]
        current = input['current']
        model = input['model']
        onehot_y = self.label2onehot(y, self.num_classes)
        x,y, onehot_y  = x.to(self.device), y.to(self.device), onehot_y.to(self.device)
        u_x = [i.to(self.device) for i in u_x]
        bt = x.size(0)
        with torch.no_grad():
            y_hat = sum([model(x)[0].log_softmax(1) for x in u_x]) / self.K
            y_hat = self.sharpen(y_hat)
            y_hat = y_hat.detach()

        # mixup
        all_inputs = torch.cat([x]+u_x, dim=0)
        all_targets = torch.cat([onehot_y] +[y_hat]*self.K, dim=0)

        lam = self.lamda.sample().item()
        lam = max(lam, 1-lam)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lam * input_a + (1 - lam) * input_b
        mixed_target = lam * target_a + (1 - lam) * target_b
        mixed_input = list(torch.split(mixed_input, bt))
        mixed_input = mixmatch_interleave(mixed_input, bt)

        logit, _ = model(mixed_input[0])
        logits = [logit]
        for j in mixed_input[1:]:
            logit, _ = model(j)
            logits.append(logit)
        logits = mixmatch_interleave(logits, bt)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        loss_x, loss_u, w = self.cal_loss(logits_x, y, logits_u, mixed_target[bt:], current, self.lambda_u)
        return loss_x, loss_u, w 

        



        
    