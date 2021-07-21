import os
import torch
from abc import ABC, abstractmethod
from utils import create_logger

class BaseModel(ABC):
    def __init__(self, configs):
        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger, writer, out_dir = create_logger(self.configs)
        self.logger = logger
        self.writer = writer
        self.out_dir = out_dir

    def validation(self):
        """
        Validating the model while training
        """
        pass

    def train(self):
        """
        Training the model
        """
        pass
    
    def evaluate(self):
        """
        Evaluating the model 
        """
        pass
    
    def _load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.logger.info(f"  Loading the checkpoint from {ckpt_path}")
        for key in checkpoint:
            try:
                getattr(self, key).load_state_dict(checkpoint[key])
            except:
                setattr(self, key, checkpoint[key])
        self.logger.info(f"  Loading Done.. {ckpt_path}")
        self.logger.info("  top1_ema_val : {:.2f}".format(self.top1_ema_val))
        self.logger.info("  Training resumes from epoch {0}".format(checkpoint['epoch']))
        return checkpoint['epoch']

    def _save_checkpoint(self, epoch, is_best_val=False):
        # latest checkpoint
        model        = getattr(self, 'model')
        ema_model    = getattr(self, 'ema_model')
        optimizer    = getattr(self, 'optimizer')
        lr_scheduler = getattr(self, 'lr_scheduler')
        top1_val     = getattr(self, 'top1_val')
        top1_ema_val = getattr(self, 'top1_ema_val')

        checkpoint = dict()
        checkpoint['model'] = model.state_dict()
        checkpoint['ema_model'] = ema_model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        checkpoint['top1_val'] = top1_val
        checkpoint['top1_ema_val'] = top1_ema_val
        checkpoint['epoch'] = epoch + 1
        if is_best_val:
            torch.save(checkpoint, os.path.join(self.out_dir, 'best.pth'))  
            self.logger.info("  Best saving ... ")
        else:
            torch.save(checkpoint, os.path.join(self.out_dir, 'latest.pth'))