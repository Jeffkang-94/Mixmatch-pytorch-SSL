from model import *
from config import *
from utils import *
from src import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def main():
    args = parse_args()
    configs = get_configs(args)
    if configs.mode == 'train':
        MixmatchTrainer = Trainer(configs)
        MixmatchTrainer.train()
    elif configs.mode == 'eval':
        MixmatchEvaluator = Evaluator(configs)
        MixmatchEvaluator.evaluate()
    else:
        raise ValueError ("Invalid mode, ['train', 'eval'] modes are supported")
        
if __name__ == '__main__':
    main()
