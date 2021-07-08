# PyTorch-MixMatch - A Holistic Approach to Semi-Supervised Learning

:warning: Unofficial reproduced code for **[MixMatch](https://arxiv.org/pdf/1905.02249.pdf)** by PyTorch.
This repository covers a variety of dataset e.g., CIFAR-10, CIFAR-100, STL-10, MiniImageNet, etc.

- [X] *2021-07-08* Implementing Mixmatch using CIFAR-10 dataset.
- [ ] Evaluation code
- [ ] Supporting other datasets
- [ ] Upload Pre-trained model and Experimental results
- [ ] Trouble shooting

## :hammer: Setup

### Dependency

```
pytorch > 1.0
torchvision > 0.5
tqdm
tensorboardX > 2.0
```

### Dataset

You have to specify the datapath using symbolic link or directly download the corresponding dataset under the `data` folder.

```bash
> mkdir data  
> ln -s ${datapath} data
```


## :rainbow: Training
To train MixMatch model, just follow the below command with a configuration file.

```bash
python main.py --cfg_path configs/${config_name}
```

Training configurations are located under `config` folder. You can tune the each parameter.
MixMatch has 4 primary parameter: `lambda_u, alpha, T` and ` K`. (See 3.5 section of [MixMatch](https://arxiv.org/pdf/1905.02249.pdf))
The original paper fixes the `T` and `K` as `0.5` and `2`, respectively.
The authors vary the value of `lambda_u` and `alpha` depending on the type of dataset.
CIFAR-10, for instance, `lambda_u=75` and `alpha=0.5` are used.
Specifically, they mentioned that *`lambda_u=100` and `alpha=0.75` are good starting points for tunning*.
For those who want to use a custom dataset, you can refer to that mention.
This is an example configuration for CIFAR-10 dataset.

```python
{
    "mode": "train",        # mode [train/eval]
    "name": "Mixmatch",     # name of trial
    "dataset": "CIFAR10",   # dataset [CIFAR10, CIFAR100, STL-10, SVHN]
    "datapath":"./data",    # datapath
    "depth":28,             # ResNet depth
    "width":2,              # ResNet width
    "num_classes":10,       # Number of class, e.g., CIFAR-10 : 10
    "num_label":250,        # The number of available label [250, 1000, 4000]
    "batch_size":64,        # batch size
    "epochs":1024,          # epoch
    "save_epoch":1,         # interval of saving checkpoint
    "resume": false,        # resuming the training

    /* Training Configuration */
    "lr":3e-2,              
    "lambda_u": 75,         
    "alpha":0.75,           
    "T" : 0.5,              # fixed across all experiments, but you can adjust it
    "K" : 2,                # fixed across all experiments, but you can adjust it
    "ema_alpha":0.999,
    "weight_decay":0.0004,
    "seed":3114
}
```

 - `lambda_u` : A hyper-parameter weighting the contribution of the unlabeled examples to the training loss
 - `alpha`    : Hyperparameter for the Beta distribution used in MixU
 - `T`        : Temperature parameter for sharpening used in MixMatch
 - `K`        : Number of augmentations used when guessing labels in MixMatch

### Example

Training MixMatch on WideResNet28x2 using a CIFAR10 with 250 labeled data

> python main.py --cfg_path config/train_CIFAR10_250.json

## :gift: Pre-training model

/* not supported yet */

## :link: Experiments

### Table

**CIFAR-10** | 250  | 500 | 1000 | 2000 | 4000 |
| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
#Paper | 88.92±0.87	| 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 | 
**#This repo** | 0 | 0 | 0 | 0 | 0  | 0 | 

**SVHN** | 250  | 500 | 1000 | 2000 | 4000 |
| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
#Paper | 96.22±0.87	| 96.36±0.94 | 96.73±0.32 | 96.96±0.15 | 97.11±0.06 | 
**#This repo** | 0 | 0 | 0 | 0 | 0  | 0 | 

### Training log


## Reference

YU1ut [MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch)
perrying [realistic-ssl-evaluation-pytorch](https://github.com/perrying/realistic-ssl-evaluation-pytorch)