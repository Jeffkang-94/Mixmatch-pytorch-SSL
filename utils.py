import logging
import os,sys
import torch
from tensorboardX import SummaryWriter
import data_loader.transform as T
from data_loader.randaugment import RandAugmentMC as RandomAugment
import torchvision.transforms as transforms
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

def get_fixmatch_transform(_dataset):
    mean, std = get_normalize(_dataset)
    if _dataset=='CIFAR10' or _dataset=='CIFAR100' or _dataset=='SVHN':
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandomAugment(n=2, m=10), # RandomAugmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return (weak, strong), transform_val

def get_normalize(_dataset):
    if _dataset == 'CIFAR10':
        return (0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616)
    elif _dataset =='CIFAR100':
        return (0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)
    elif _dataset =='SVHN':
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    elif _dataset =='STL10':
        return (0.4409, 0.4279, 0.3868), (0.2683, 0.2611, 0.2687)
    else:
        raise NotImplementedError

def get_mixmatch_transform(_dataset):
    mean, std = get_normalize(_dataset)
    if _dataset=='CIFAR10' or _dataset=='CIFAR100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    elif _dataset == 'SVHN':
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    elif _dataset =='STL10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=int(96*0.125), padding_mode='reflect'),
            transforms.RandomFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise NotImplementedError

    return train_transform, test_transform

def get_transform(method, _dataset):
    if method == 'Mixmatch':
        return get_mixmatch_transform(_dataset)
    elif method =='Fixmatch':
        return get_fixmatch_transform(_dataset)
    else:
        raise NotImplementedError

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
    

def interleave(x, bt):
    # bt: number of batches of labeled data
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])



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

def create_logger(configs):
    result_dir = os.path.join("results", configs.name) # "results/MixMatch"
    os.makedirs(result_dir, exist_ok=True)
    out_dir = os.path.join(result_dir, str(configs.dataset) + '_' + str(configs.depth) + '-' +str(configs.width) + '_' + str(configs.num_label))
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(result_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    log_file = '{}.log'.format(configs.name)
    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(head))
    logger.addHandler(console_handler)

    if configs.mode =='train':
        if configs.method =='Mixmatch':
            logger.info(f"  Desc        = PyTorch Implementation of MixMatch")
            logger.info(f"  Task        = {configs.dataset}@{configs.num_label}")
            logger.info(f"  Model       = WideResNet {configs.depth}x{configs.width}")
            logger.info(f"  large model = {configs.large}")
            logger.info(f"  Batch size  = {configs.batch_size}")
            logger.info(f"  Epoch       = {configs.epochs}")
            logger.info(f"  Optim       = {configs.optim}")
            logger.info(f"  lambda_u    = {configs.lambda_u}")
            logger.info(f"  alpha       = {configs.alpha}")
            logger.info(f"  T           = {configs.T}")
            logger.info(f"  K           = {configs.K}")
        elif configs.method =='Fixmatch':
            logger.info(f"  Desc        = PyTorch Implementation of FixMatch")
            logger.info(f"  Task        = {configs.dataset}@{configs.num_label}")
            logger.info(f"  Model       = WideResNet {configs.depth}x{configs.width}")
            logger.info(f"  large model = {configs.large}")
            logger.info(f"  Batch size  = {configs.batch_size}")
            logger.info(f"  Epoch       = {configs.epochs}")
            logger.info(f"  Optim       = {configs.optim}")
            logger.info(f"  lambda_u    = {configs.lambda_u}")
            logger.info(f"  threshold   = {configs.threshold}")
            logger.info(f"  K           = {configs.K}")
            logger.info(f"  mu          = {configs.mu}")


    return logger, writer, out_dir
