# This shell file will generate the experimental results of Appendix B: B.1, B.2

# CIFAR10
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_250.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_500.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_1000.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_2000.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_4000.json

#SVHN
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/SVHN/train_SVHN_250.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/SVHN/train_SVHN_500.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/SVHN/train_SVHN_1000.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/SVHN/train_SVHN_2000.json
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path config/mixmatch/SVHN/train_SVHN_4000.json