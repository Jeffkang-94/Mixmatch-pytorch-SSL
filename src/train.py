import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
import time

from data_loader.loader import get_trainval_data
from src.base import BaseModel
from SSL_loss.mixmatch import MixMatchLoss
from tqdm import tqdm
from utils import *
from model import *

class Trainer(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model = WRN(num_classes=configs.num_classes, 
                        depth=configs.depth, 
                        width=configs.width,
                        large=configs.large).to(self.device)
        
        self.ema_model  = WRN(num_classes=configs.num_classes, 
                        depth=configs.depth, 
                        width=configs.width,
                        large=configs.large).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()
        
        if self.configs.seed == "None":
            manualSeed = random.randint(1, 10000)
        else:
            manualSeed = self.configs.seed
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        self.logger.info("  Total params: {:.2f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6))
        self.logger.info("  Sampling seed : {0}".format(manualSeed))
        transform_train, transform_val = get_transform(configs.method, configs.dataset)
        
        train_labeled_set, train_unlabeled_set, val_set  = get_trainval_data(configs.datapath, configs.method, configs.dataset, configs.K, \
            configs.num_label, configs.num_classes, transform_train=transform_train, transform_val=transform_val)
        
        if configs.method=='Mixmatch':
            self.train_loader = data.DataLoader(train_labeled_set, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
            self.u_train_loader = data.DataLoader(train_unlabeled_set, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
        else:
            raise NotImplementedError
        self.val_loader = data.DataLoader(val_set, batch_size=configs.batch_size, shuffle=False, num_workers=0)

        self.criterion      = MixMatchLoss(configs, self.device)
        self.eval_criterion = nn.CrossEntropyLoss().to(self.device)
        if configs.optim == 'ADAM':
            self.optimizer      = torch.optim.Adam(self.model.parameters(), lr = configs.lr)
        elif configs.optim =='SGD':
            self.optimizer      = torch.optim.SGD(self.model.parameters(), lr = configs.lr, momentum=0.9, nesterov=True, weight_decay=configs.weight_decay)
        self.ema_optimizer  = WeightEMA(self.model, self.ema_model, configs.weight_decay*configs.lr, alpha=configs.ema_alpha)

        self.top1_val        = 0 
        self.top1_ema_val    = 0

        if self.configs.resume:
            ckpt_path = os.path.join(self.out_dir, self.configs.ckpt)
            self.start_epoch = self._load_checkpoint(ckpt_path)
        else:
            self.start_epoch = 0

    def _terminate(self):
        # terminate the logger and SummaryWriter
        self.writer.close()

    def evaluate(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        loss_ema_meter = AverageMeter()
        top1_ema_meter = AverageMeter()
        top5_ema_meter = AverageMeter()
        
        is_best_val = False
        ema_is_best_val = False
        with torch.no_grad():
            if self.configs.verbose:
                tq = tqdm(self.val_loader, total=self.val_loader.__len__(), leave=False)
            else:
                tq = self.val_loader
            for x,y in tq:
                
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                logits_ema, _ = self.ema_model(x)

                loss = self.eval_criterion(logits, y)
                prob = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(prob, y, (1,5))

                loss_meter.update(loss.item())
                top1_meter.update(top1.item())
                top5_meter.update(top5.item())

                loss_ema = self.eval_criterion(logits_ema, y)
                prob_ema = torch.softmax(logits_ema, dim=1)
                top1_ema, top5_ema = accuracy(prob_ema, y, (1,5))

                loss_ema_meter.update(loss_ema.item())
                top1_ema_meter.update(top1_ema.item())
                top5_ema_meter.update(top5_ema.item())
                if self.configs.verbose:
                    tq.set_description("[{}] Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format("VAL", epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))
        self.logger.info("  [{}] Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format(" VAL ", epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))
        self.logger.info("  [{}] Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format(" EMA ", epoch, top1_ema_meter.avg, top5_ema_meter.avg, loss_ema_meter.avg))
        self.writer.add_scalars('val_acc/top1', {
            'top1_val': top1_meter.avg,
            'top1_ema_val': top1_ema_meter.avg,
        }, epoch)
        self.writer.add_scalars('val_acc/top5', {
            'top5_val': top5_meter.avg,
            'top5_ema_val': top5_ema_meter.avg
        }, epoch)
        self.writer.add_scalars('val_loss', {
            'loss_val': loss_meter.avg,
            'loss_ema_val': loss_ema_meter.avg
        }, epoch)

        if self.top1_val < top1_meter.avg:
            self.top1_val = top1_meter.avg
            is_best_val = True
        if self.top1_ema_val < top1_ema_meter.avg:
            self.top1_ema_val = top1_ema_meter.avg
            ema_is_best_val = True
        self._save_checkpoint(epoch, is_best_val, ema_is_best_val)
        

    def train(self):
        
        epoch_start = time.time()  # start time
        loss_meter   = AverageMeter()
        loss_x_meter = AverageMeter()
        loss_u_meter = AverageMeter()
        n_iters = 1024
        train_loader_iter  = iter(self.train_loader)
        u_train_loader_iter = iter(self.u_train_loader)
        for epoch in range(self.start_epoch, self.configs.epochs):
            self.model.train()
            if self.configs.verbose:
                tq = tqdm(range(n_iters), total = n_iters, leave=True)
            else:
                tq = range(n_iters)
            for it in tq:
                try:
                    x, y = train_loader_iter.next()
                except:
                    train_loader_iter  = iter(self.train_loader)
                    x, y = train_loader_iter.next()
                try:
                    u_x, _ = u_train_loader_iter.next()
                except:
                    u_train_loader_iter = iter(self.u_train_loader)
                    u_x, _  = u_train_loader_iter.next()

                # forward inputs
                
                current = epoch + it / n_iters
                input = {'model'    : self.model, 
                         'u_x'      : u_x, 
                         'x'        : x, 
                         'y'        : y,
                         'current'  : current}

                # compute mixmatch loss
                loss_x, loss_u, w = self.criterion(input)
                loss = loss_x + loss_u * w

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_optimizer.step()
                

                # logging
                loss_meter.update(loss.item())
                loss_x_meter.update(loss_x.item())
                loss_u_meter.update(loss_u.item())

                t = time.time() - epoch_start
                if self.configs.verbose:
                    tq.set_description(" Epoch [{}/{}], iter: {}. loss: {:.4f}. loss_x: {:.4f}. loss_u: {:.4f}.  weight :{:.4f} Time: {:.2f}".format(
                        epoch, self.configs.epochs, it + 1, 
                        loss_meter.avg, 
                        loss_x_meter.avg, 
                        loss_u_meter.avg,
                        w, 
                        t))
            self.logger.info(" Epoch [{}/{}], iter: {}. loss: {:.4f}. loss_x: {:.4f}. loss_u: {:.4f}. weight : {:.4f}. Time: {:.2f}".format(
                    epoch, self.configs.epochs, it + 1, 
                    loss_meter.avg, 
                    loss_x_meter.avg, 
                    loss_u_meter.avg,
                    w,
                    t))
            self.writer.add_scalars('train_loss', {
                'loss': loss_meter.avg,
                'loss_x': loss_x_meter.avg,
                'loss_u': loss_u_meter.avg,
            }, epoch)   
            self.evaluate(epoch)
        self._terminate()
        


