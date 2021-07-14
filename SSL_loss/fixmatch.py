import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

class FixMatchLoss(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.K = configs.K
        self.T = configs.T
        self.bt = configs.batch_size
        self.epochs = configs.epochs
        self.lambda_u = configs.lambda_u
        self.beta_dist = torch.distributions.beta.Beta(configs.alpha, configs.alpha)
        self.num_classes = configs.num_classes
        self.device = device


    def cal_loss(self, logit_x, y, logit_u_x, y_hat, epoch, lambda_u):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

        



        
    