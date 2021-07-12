import torch
import copy
import os
from model import *
import torch.utils.data as data
import data_loader.transform as T
from data_loader.loader import get_test_data
from tqdm import tqdm
from src.base import BaseModel
from utils import *

class Evaluator(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model = WRN(num_classes=configs.num_classes, 
                        depth=configs.depth, 
                        width=configs.width,
                        large=configs.large).to(self.device)
        self.ema_model      = copy.deepcopy(self.model).to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()

        transform_val = T.Compose([
            T.ToTensor(),
        ])
        test_set = get_test_data(self.configs.datapath, self.configs.dataset, transform_val)
        self.test_loader = data.DataLoader(test_set, batch_size=configs.batch_size, shuffle=False, num_workers=0, drop_last=False)
        self.eval_criterion = nn.CrossEntropyLoss().to(self.device)

        ckpt_path = os.path.join(self.out_dir, self.configs.ckpt)
        self._load_checkpoint(ckpt_path)


    
    def evaluate(self):
        self.model.eval()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        loss_ema_meter = AverageMeter()
        top1_ema_meter = AverageMeter()
        top5_ema_meter = AverageMeter()
        
        with torch.no_grad():
            tq = tqdm(self.test_loader, total=self.test_loader.__len__(), leave=False)
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
                tq.set_description("[{}] Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format("TEST", top1_meter.avg, top5_meter.avg, loss_meter.avg))
        self.logger.info("  [{}] Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format("TEST", top1_meter.avg, top5_meter.avg, loss_meter.avg))
        self.logger.info("  [{}] Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format("EMA ", top1_ema_meter.avg, top5_ema_meter.avg, loss_ema_meter.avg))