import os
import time
import shutil
import random
import torch
import numpy as np
from model import *
from loss import MixMatchLoss
from config import *
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
import data_loader.transform as T
from data_loader.cifar10 import get_cifar10

def train_one_epoch(epoch,
                    model,
                    criterion,
                    optim,
                    lr_schdlr,
                    ema,
                    dltrain_x,
                    dltrain_u,
                    lambda_u,
                    n_iters,
                    alpha,
                    T,
                    n_class
                    ):

    model.train()
    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()

    tq = tqdm(range(n_iters), total = n_iters, leave=True)
    for it in tq:
        ims_x, _,targets_x = next(dl_x)
        ims_u1, ims_u2, _ = next(dl_u)

        bt = ims_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(bt, n_class).scatter_(1, targets_x.view(-1,1).long(), 1)

        ims_x, targets_x = ims_x.cuda(), targets_x.cuda()
        ims_u1, ims_u2 = ims_u1.cuda(), ims_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u1, _ = model(ims_u1)
            outputs_u2, _ = model(ims_u2)
            p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        #print(targets_x.shape, targets_u.shape, targets_u.shape)
        all_inputs = torch.cat([ims_x, ims_u1, ims_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        lamda = np.random.beta(alpha, alpha)
        lamda = max(lamda, 1-lamda)

        newidx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[newidx]
        target_a, target_b = all_targets, all_targets[newidx]

        mixed_input = lamda * input_a + (1 - lamda) * input_b
        mixed_target = lamda * target_a + (1 - lamda) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, bt))
        mixed_input = mixmatch_interleave(mixed_input, bt)

        logit, _ = model(mixed_input[0])
        logits = [logit]
        for input in mixed_input[1:]:
            logit, _ = model(input)
            logits.append(logit)

         # put interleaved samples back
        logits = mixmatch_interleave(logits, bt)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        loss_x, loss_u, w = criterion(logits_x, mixed_target[:bt], logits_u, mixed_target[bt:], epoch+it/n_iters, lambda_u)

        loss = loss_x + w * loss_u
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        t = time.time() - epoch_start
        tq.set_description("Epoch:{}, iter: {}. loss: {:.4f}. loss_x: {:.4f}. loss_u: {:.4f}. lambda_u: {:.4f}. Time: {:.2f}".format(
                epoch, it + 1, 
                loss_meter.avg, 
                loss_x_meter.avg, 
                loss_u_meter.avg,
                w, 
                t))
        epoch_start = time.time()

    ema.update_buffer()

 
    return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg

def evaluate(epoch, model, dataloader, criterion):
    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    tq = tqdm(dataloader, total=len(dataloader), leave=True)
    with torch.no_grad():
        #for ims, lbs in dataloader:
        for ims, lbs in tq:
            ims = ims.cuda()
            lbs = lbs.cuda()
            logits, _ = model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

    tq.set_description("Test Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format(epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))

    return top1_meter.avg, top5_meter.avg, loss_meter.avg

def evaluate_ema(epoch, ema, dataloader, criterion):
    # using EMA params to evaluate performance
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    tq = tqdm(dataloader, total=len(dataloader), leave=True)
    with torch.no_grad():
       # for ims, lbs in dataloader:
       for ims, lbs in tq:
            ims = ims.cuda()
            lbs = lbs.cuda()
            logits, _ = ema.model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
    # note roll back model current params to continue training
    ema.restore()

    tq.set_description("Test Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format(epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))
    return top1_meter.avg, top5_meter.avg, loss_meter.avg
  
def main():
    args = parse_args()
    configs = get_configs(args)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create folder for training and copy important files
    result_dir = os.path.join("results", configs.name) # "results/MixMatch"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    out_dir = os.path.join(result_dir, "trial")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #else:
   #     print('Error: the folder name is dupplicated!')
     #   return
    
    #args.n_iters_per_epoch = args.k_imgs // args.batchsize  
    #args.n_iters_all = args.n_iters_per_epoch * args.epochs  
    #args.n_classes, args.num_val = 10, 5000
 
    model = WRN(num_classes=configs.num_label, depth=configs.depth, width=configs.width).to(device)
    
    criterion = MixMatchLoss()
    dltrain_x, dltrain_u, dlval = get_data(configs)

    ema = EMA(model, configs.ema_alpha)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias
            # print(name)
        else:
            wd_params.append(param)
    #param_list = [
    #    {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay, momentum=0.9, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(
        optim, max_iter=1024*configs.epochs, warmup_iter=0
    )

    best_acc_val = -1
    best_epoch_val = 0
    best_acc_ema_val = -1
    best_epoch_ema_val = 0

    best_acc = -1
    best_epoch = 0
    best_acc_ema = -1
    best_epoch_ema = 0


    train_args = dict(
        model=model,
        criterion=criterion,
        optim=optim,
        lr_schdlr=lr_schdlr,
        ema=ema,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        lambda_u=configs.lambda_u,
        n_iters=1024,
        alpha=configs.alpha,
        T=configs.T,
        n_class=configs.num_label
    )

    #global num_epochs
    num_epochs = configs.epochs
    start_epoch = 0 
    criterion = nn.CrossEntropyLoss().to(device)


    for epoch in range(start_epoch, configs.epochs):
        train_loss, loss_x, loss_u = train_one_epoch(epoch, **train_args)

        top1_val, top5_val, valid_loss_val = evaluate(epoch, model, dlval, criterion)
        top1_ema_val, top5_ema_val, valid_loss_ema_val = evaluate_ema(epoch, ema, dlval, criterion)

        is_best_val = best_acc_val < top1_val
        if is_best_val:
            best_acc_val = top1_val
            best_epoch_val = epoch

        is_best_ema_val = best_acc_ema_val < top1_ema_val
        if is_best_ema_val:
            best_acc_ema_val = top1_ema_val
            best_epoch_ema_val = epoch

        torch.save({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.shadow,   # not ema.model.state_dict()
            'top1_val': top1_val,
            'best_top1_val': best_acc_val,
            'best_epoch_val': best_epoch_val,
            'top1_ema_val': top1_ema_val,
            'best_top1_ema_val': best_acc_ema_val,
            'best_epoch_ema_val': best_epoch_ema_val,
            'optimizer': optim.state_dict(),
        }, os.path.join(result_dir + '_checkpoint'))

      
        if (epoch + 1) % configs.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, configs.name + '_e{}'.format(epoch)))
            torch.save(ema.shadow, os.path.join(out_dir, configs.name + '_ema_e{}'.format(epoch)))    # not ema.model.state_dict()

if __name__ == '__main__':
    main()
