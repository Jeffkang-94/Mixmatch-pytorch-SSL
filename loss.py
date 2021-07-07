import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np
from utils import *

class MixMatchLoss(nn.Module):
    def __init__(self, alpha, lambda_u, T, K):
        super().__init__()
        self.K = K
        self.T = T
        self.lambda_u = lambda_u
        self.lamda = Beta(alpha, alpha)

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

    def cal_loss(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = F.mse_loss(probs_u, targets_u)

        return Lx, Lu, lambda_u * self.linear_rampup(epoch)  

    def forward(self, input):
        bt = input['x'].size(0)
        with torch.no_grad():
            logit_u1, _ = input['model'](input['u_x_1'])
            logit_u2, _ = input['model'](input['u_x_2'])
            y_hat = self.sharpen(y_hat)
            y_hat = y_hat.detach()

        # mixup
        all_inputs = torch.cat([x, x_u1, x_u2], dim=0)
        all_targets = torch.cat([y, y_hat, y_hat], dim=0)

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
        for input in mixed_input[1:]:
            logit, _ = model(input)
            logits.append(logit)
        logits = mixmatch_interleave(logits, bt)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        loss_x, loss_u, w = self.cal_loss(logits_x, mixed_target[:bt], logits_u, mixed_target[bt:], current, self.lambda_u)
        return loss_x, loss_u, w 

        



        
    