import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Learning')

    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')

    parser.add_argument('--run_dir', type=str, default='run', help='the directory for training')
    parser.add_argument('--name',default='Mix_YU1ut', type=str, help='output model name')
    parser.add_argument('--log_interval', type=int, default=10, help='log training status')
    parser.add_argument('--save_epoch', type=int, default=64, help='step to save model')

    parser.add_argument('--resume',default=None, type=str, help='checkpoint to continue')

    parser.add_argument('--backbone', type=str, default='wideresnet', help='Wideresnet')
    parser.add_argument('--wresnet-k', default=2, type=int, help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int, help='depth of wide resnet')
    parser.add_argument('--large_model', action='store_true', help='default is False. If True, using WideResnetLarge model')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10, CIFAR100, SVHN, or STL10')
    parser.add_argument('--num_label', type=int, default=40, help='number of labeled samples for training')
    parser.add_argument('--num_val', type=int, default=5000, help='number of samples of cross-validation set')

    parser.add_argument('--fold', type=int, default=0, help='used for STL10. This is to pick respective 1000-examples fold')

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training')
    parser.add_argument('--epochs', type=int, default=1024, help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64, help='train batch size of labeled samples')
    parser.add_argument('--valbatchsize', default=100, type=int, help='batchsize')
    parser.add_argument('--mu', type=int, default=1, help='factor of train batch size of unlabeled samples')
        
    parser.add_argument('--thr', type=float, default=0.95, help='pseudo label threshold')

    parser.add_argument('--k_imgs', type=int, default=64 * 1024, help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=75, help='coefficient of unlabeled loss')
    parser.add_argument('--ema-alpha', type=float, default=0.999, help='decay rate for ema module')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')

    parser.add_argument('--alpha', default=0.75, type=float, help='for Mixup algorithm')
    parser.add_argument('--T', default=0.5, type=float, help='used for sharpening, entropy minimization')

    parser.add_argument('--seed', type=int, default=-1, help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--seed_init', type=int, default=10000, help='the number to feed to seed()')

    parser.add_argument('--cudnn_deter', action='store_false', help='default is True. If True, use deterministic functions as much as possible')
    parser.add_argument('--cudnn_bench', action='store_true', help='default is False. If True, program may run faster')
    
    return parser.parse_args()