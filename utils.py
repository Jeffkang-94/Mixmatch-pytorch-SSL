import logging
import os
import shutil
import torch
from tensorboardX import SummaryWriter

def mixmatch_interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def mixmatch_interleave(xy, batch):
    nu = len(xy) - 1
    offsets = mixmatch_interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
    
def get_data(configs):

    assert configs.dataset in ['CIFAR10', 'CIFAR100', 'SVHN', 'STL10', 'TinyImageNet']

    if configs.dataset == 'CIFAR10' or configs.dataset == 'CIFAR100':
        from data_loader.cifar import get_train_loader_mixmatch
    elif configs.dataset == 'SVHN':
        from data_loader.svhn import get_train_loader_mixmatch
    elif configs.dataset == 'STL10':
        from data_loader.stl10 import get_train_loader_mixmatch
    elif configs.dataset == 'TinyImageNet':
        from data_loader.tiny_imagenet import get_train_loader_mixmatch
    train_loader, val_loader = get_train_loader_mixmatch(
        configs.dataset, configs.K, configs.batch_size, 1024, configs.num_label, configs.datapath)

    return train_loader, val_loader
    
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

def logger(configs):
    # Create folder for training and copy important files
    result_dir = os.path.join("results", configs.name) # "results/MixMatch"
    os.makedirs(result_dir, exist_ok=True)
    out_dir = os.path.join(result_dir, str(configs.dataset) + '_' + str(configs.depth) + '-' +str(configs.width) + '_' + str(configs.num_label))
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(result_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, out_dir


class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]

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
