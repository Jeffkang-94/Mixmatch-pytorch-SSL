import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

class FixMatchLoss(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.threshold = configs.threshold
        self.bt = configs.batch_size
        self.lambda_u = configs.lambda_u
        self.num_classes = configs.num_classes
        self.mu = configs.mu 
        self.device = device
        self.T = configs.T


    def cal_loss(self, logits_x, y , logits_u_s, y_hat, mask):
        loss_x = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * y, dim=1)) # Cross entropy 
        loss_u = (F.cross_entropy(logits_u_s, y_hat, reduction='none')*mask).mean()
        return loss_x, loss_u, self.lambda_u

    def forward(self, input):
        x = input['x']
        y = input['y']
        u_x = [x for x in input['u_x']]
        model = input['model']
        
        # make onehot label
        #y = F.one_hot(y, self.num_classes)
        x,y  = x.to(self.device), y.to(self.device)
        #u_x = [i.to(self.device) for i in u_x]
        w_x, s_x = u_x[0].to(self.device), u_x[1].to(self.device)
        
        inputs = interleave(
                torch.cat((x, w_x, s_x)), 2*self.mu+1).to(self.device)
        y = y.to(self.device)
        logits = model(inputs)[0]
        logits = de_interleave(logits, 2*self.mu+1)
        logits_x = logits[:self.bt]
        logits_u_w, logits_u_s = logits[self.bt:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, y)

        pseudo_label = torch.softmax(logits_u_w.detach()/self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask).mean()
        return Lx, Lu, self.lambda_u



        
    
