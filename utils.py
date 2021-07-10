import logging
import os,sys
import torch
from tensorboardX import SummaryWriter

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

    return logger, writer, out_dir
