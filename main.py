import os
import time
import random
import torch
import numpy as np
from model import *
from loss import MixMatchLoss
from config import *
from utils import *
from tqdm import tqdm
import data_loader.transform as T
import torch.utils.data as data
from data_loader.loader import get_cifar10
import torch.backends.cudnn as cudnn

def train_one_epoch(epoch,
                    model,
                    criterion,
                    optim,
                    lr_schdlr,
                    ema_optimizer,
                    train_loader,
                    u_train_loader,
                    n_iters,
                    ):

    model.train()
    epoch_start = time.time()  # start time
    
    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    tq = tqdm(range(n_iters), total = n_iters, leave=True)
    for it in tq:
        try:
            x, y = train_loader_iter.next()
        except:
            train_loader_iter  = iter(train_loader)
            x, y = train_loader_iter.next()
        try:
            u_x, _ = u_train_loader.next()
        except:
            u_train_loader_iter = iter(u_train_loader)
            u_x, _  = u_train_loader_iter.next()
        current = epoch + it / n_iters
        input = {'model'    : model, 
                 'u_x'    : u_x, 
                 'x'        : x, 
                 'y'        : y,
                 'current'  : current}

        loss_x, loss_u, w = criterion(input)
        loss = loss_x + w * loss_u
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema_optimizer.step()
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

    #ema_model.update_buffer()
    return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg

def evaluate(epoch, model, dataloader, criterion, device):
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    tq = tqdm(dataloader, total=len(dataloader), leave=True)
    with torch.no_grad():
        for ims, lbs in tq:
            ims = ims.to(device)
            lbs = lbs.to(device)
            logits, _ = model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())
            tq.set_description("Test Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format(epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))
    logger.info("  [{}] Test Epoch:{}. Top1: {:.4f}. Top5: {:.4f}. Loss: {:.4f}.".format("EVAL", epoch, top1_meter.avg, top5_meter.avg, loss_meter.avg))

    return top1_meter.avg, top5_meter.avg, loss_meter.avg
  
def main():
    args = parse_args()
    configs = get_configs(args)
    cudnn.benchmark = True
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global logger
    logger, writer, out_dir = create_logger(configs)
    
    model = WRN(num_classes=configs.num_classes, 
                depth=configs.depth, 
                width=configs.width,
                large=configs.large).to(device)
    ema_model = WRN(num_classes=configs.num_classes, 
                depth=configs.depth, 
                width=configs.width,
                large=configs.large).to(device)

    for param in ema_model.parameters():
        param.detach_()


    criterion = MixMatchLoss(configs.alpha, 
                    configs.lambda_u, 
                    configs.T,
                    configs.K, 
                    configs.num_classes, 
                    device)
    transform_train = T.Compose([
        T.RandomPadandCrop(32),
        T.RandomFlip(),
        T.ToTensor(),
    ])

    transform_val = T.Compose([
        T.ToTensor(),
    ])
    train_labeled_set, train_unlabeled_set, val_set, test_set = get_cifar10('./data', configs.K, configs.num_label, transform_train=transform_train, transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=configs.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=configs.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=configs.batch_size, shuffle=False, num_workers=0)

    ema_optimizer = WeightEMA(model, ema_model, alpha=configs.ema_alpha)

    #optim = torch.optim.SGD(model.parameters(), lr=configs.lr, momentum=0.9)#, weight_decay=configs.weight_decay, momentum=0.9, nesterov=True)
    optim = torch.optim.Adam(model.parameters(), lr = configs.lr, weight_decay=configs.weight_decay)
    lr_schdlr = WarmupCosineLrScheduler(
        optim, max_iter=1024*configs.epochs, warmup_iter=0
    )
    logger.info("  Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    best_acc_val = -1
    best_epoch_val = 0
    best_acc_ema_val = -1
    best_epoch_ema_val = 0

    #global num_epochs
    start_epoch = 0 
    CE_loss = nn.CrossEntropyLoss().to(device)


    train_args = dict(
        model=model,
        criterion=criterion,
        optim=optim,
        lr_schdlr=lr_schdlr,
        ema_optimizer=ema_optimizer,
        train_loader=labeled_trainloader,
        u_train_loader = unlabeled_trainloader,
        n_iters=1024
    )

    logger.info(f"  Start trainng")
    for epoch in range(start_epoch, configs.epochs):
        train_loss, loss_x, loss_u = train_one_epoch(epoch, **train_args)
        top1_val, top5_val, valid_loss_val = evaluate(epoch,  model, val_loader, CE_loss, device)
        top1_ema_val, top5_ema_val, valid_loss_ema_val = evaluate(epoch, ema_model, val_loader, CE_loss, device)
        top1_test, top5_test, valid_loss_test = evaluate(epoch,  model, test_loader, CE_loss, device)
        top1_ema_test, top5_ema_test, valid_loss_ema_test = evaluate(epoch, ema_model, test_loader, CE_loss, device)
        
        # train log
        writer.add_scalars('train_loss', {
            'loss': train_loss,
            'loss_x': loss_x,
            'loss_u': loss_u
        }, epoch)

        # validation log
        writer.add_scalars('val_acc/top1', {
            'top1_val': top1_val,
            'top1_ema_val': top1_ema_val,
        }, epoch)
        writer.add_scalars('val_acc/top5', {
            'top5_val': top5_val,
            'top5_ema_val': top5_ema_val
        }, epoch)
        writer.add_scalars('val_loss', {
            'loss_val': valid_loss_val,
            'loss_ema_val': valid_loss_ema_val
        }, epoch)

        # test log
        writer.add_scalars('test_acc/top1', {
            'top1_test': top1_test,
            'top1_ema_test': top1_ema_test,
        }, epoch)
        writer.add_scalars('test_acc/top5', {
            'top5_test': top5_test,
            'top5_ema_test': top5_ema_test
        }, epoch)
        writer.add_scalars('test_loss', {
            'loss_test': valid_loss_test,
            'loss_ema_test': valid_loss_ema_test
        }, epoch)


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
            'model': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),   
            'top1_val': top1_val,
            'best_top1_val': best_acc_val,
            'best_epoch_val': best_epoch_val,
            'top1_ema_val': top1_ema_val,
            'best_top1_ema_val': best_acc_ema_val,
            'best_epoch_ema_val': best_epoch_ema_val,
            'optimizer': optim.state_dict(),
        }, os.path.join(out_dir + '_checkpoint.pth'))

      
        if (epoch + 1) % configs.save_epoch == 0:
            ckpt_path = os.path.join(out_dir, configs.name + '_e{}.pth'.format(epoch)) if is_best_val else os.path.join(out_dir, configs.name + "_best")
            ema_ckpt_path = os.path.join(out_dir, configs.name + '_ema_e{}.pth'.format(epoch)) if is_best_val else os.path.join(out_dir, configs.name + "_ema_best")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(ema_model.state_dict(), ema_ckpt_path)
    writer.close()
if __name__ == '__main__':
    main()
