import os
import time
import shutil
from model import WideResNet
from loss import MixMatchLoss
from config import parse_args
from tensorboardX import SummaryWriter
from trainer import 

def main():
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Create folder for training and copy important files
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.out_dir + '/src'):
        os.makedirs(args.out_dir + '/src')

    tb_log_dir = os.path.join(args.out_dir, 'tensorboard')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    #logger = create_logger(args.out_dir, args.name.lower(), time_str) 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn_deter
    torch.backends.cudnn.benchmark = args.cudnn_bench
    torch.cuda.manual_seed(args.seed)

    args.n_iters_per_epoch = args.k_imgs // args.batchsize  
    args.n_iters_all = args.n_iters_per_epoch * args.epochs  
    args.n_classes, args.num_val = get_params(args.dataset)

    if args.dataset == 'STL10':
        args.num_val = args.fold
    
    model = WideResnet(num_classes=10, depth=28, widen_factor=2)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = MixMatchLoss()

    dltrain_x, dltrain_u, dlval = get_data(args)

    ema = EMA(model, args.ema_alpha)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        # if len(param.size()) == 1:
        if 'bn' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias
            # print(name)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(
        optim, max_iter=args.n_iters_all, warmup_iter=args.warmup
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
        lambda_u=args.lam_u,
        n_iters=args.n_iters_per_epoch,
        logger=logger,
        log_interval=args.log_interval,
        alpha=args.alpha,
        T=args.T,
        n_class=args.n_classes
    )

    global num_epochs
    num_epochs = args.epochs

    logger.info(pprint.pformat(args))

    logger.info('-----------start training--------------')
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss_x, loss_u = train_one_epoch(epoch, **train_args)
        # torch.cuda.empty_cache()

        top1_val, top5_val, valid_loss_val = evaluate(epoch, model, dlval, nn.CrossEntropyLoss().cuda(), logger)

        top1_ema_val, top5_ema_val, valid_loss_ema_val = evaluate_ema(epoch, ema, dlval, nn.CrossEntropyLoss().cuda(), logger)


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
        },
        os.path.join(args.out_dir, args.name + '_checkpoint'))

        if is_best_val:
            
            torch.save(model.state_dict(), os.path.join(args.out_dir, args.name + '_bestval'))

        if is_best_ema_val:
            
            torch.save(ema.shadow, os.path.join(args.out_dir, args.name + '_ema_bestval')) # not ema.model.state_dict()

        

        if (epoch + 1) % args.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, args.name + '_e{}'.format(epoch)))
            torch.save(ema.shadow, os.path.join(args.out_dir, args.name + '_ema_e{}'.format(epoch)))    # not ema.model.state_dict()



if __name__ == '__main__':
    main()
