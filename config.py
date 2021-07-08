import argparse
import json
import utils

def get_configs(args):
    with open(args.cfg_path, "r") as f:
        configs = json.load(f)
    arg_dict = vars(args)
    for key in arg_dict : 
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
    configs = utils.ConfigMapper(configs)
    return configs
    
def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Learning')
    parser.add_argument('--cfg_path', type=str, default='./config/train.json')
    return parser.parse_args()
    
    return parser.parse_args()