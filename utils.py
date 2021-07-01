import logging
import os
import shutil

def get_data(args):

    assert args.dataset in ['CIFAR10', 'CIFAR100', 'SVHN', 'STL10', 'TinyImageNet']

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        from datasets.cifar import get_train_loader_mixmatch
    elif args.dataset == 'SVHN':
        from datasets.svhn import get_train_loader_mixmatch
    elif args.dataset == 'STL10':
        from datasets.stl10 import get_train_loader_mixmatch
    elif args.dataset == 'TinyImageNet':
        from datasets.tiny_imagenet import get_train_loader_mixmatch

    dltrain_x, dltrain_u, dlval = get_train_loader_mixmatch(
        args.dataset, args.batchsize, args.mu, args.n_iters_per_epoch, L=args.num_label, num_val=args.num_val)

    return dltrain_x, dltrain_u, dlval

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    return datetime.today().strftime(fmt)


def create_logger(out_dir, name, time_str):

    shutil.copy2(
        os.path.basename(__file__),
        out_dir)
    for f in os.listdir('.'):
        if f.endswith('.py'):
            shutil.copy2(
                f,
                out_dir + '/src')
    folders = ['datasets', 'models']
    for folder in folders:
        if not os.path.exists(out_dir + '/src/{}'.format(folder)):
            os.makedirs(out_dir + '/src/{}'.format(folder))
        for f in os.listdir(folder):
            if f.endswith('.py'):
                shutil.copy2(
                    '{}/'.format(folder) + f,
                    out_dir + '/src/{}'.format(folder))

    log_file = '{}_{}.log'.format(name, time_str)
    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(head))
    logger.addHandler(console_handler)

    return logger
