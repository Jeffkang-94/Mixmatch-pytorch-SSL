# PyTorch-MixMatch - A Holistic Approach to Semi-Supervised Learning

:warning: Unofficial reproduced code for **[MixMatch](https://arxiv.org/pdf/1905.02249.pdf)**.
This repository covers a variety of dataset e.g., CIFAR-10, CIFAR-100, STL-10, MiniImageNet, etc.

## :hammer: Setup

### Dependency

```
pytorch > 1.0
torchvision > 0.5
tqdm
tensorboardX > 2.0
```

### Dataset

You have to specify the datapath, for example, `data` folder in this codebase.
`torchvision` will automatically download the corresponding dataset(e.g., [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/),[STL10](https://cs.stanford.edu/~acoates/stl10/)) under `data` folder if `download=True`.
Or you also can directly download the datasets under your datapath and use a symbolic link instead as below.

```bash
 mkdir data  
 ln -s ${datapath} data
```


## :rainbow: Training

We maintain the code with several configuration files.
To train MixMatch model, just follow the below command with a configuration file.

```bash
python main.py --cfg_path config/${method}/${dataset}/${config_name}
```

If you want to train the model on background, refer to the below command. Plus, we recommend you to use `verbose : false` in the configuration file.

```bash
nohup python main.py --cfg_path config/${method}/${dataset}/${config_name} &
```

Training configurations are located under `config` folder. You can tune the each parameter.
Plus, `experiments` folder includes the shell files to reproduce the results introduced in the paper.
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
    "method":"Mixmatch",    # type of SSL method [Mixmatch/Fixmatch]
    "name": "Experiment1",  # name of trial
    "dataset": "CIFAR10",   # dataset [CIFAR10, CIFAR100, STL-10, SVHN]
    "datapath":"./data",    # datapath
    "depth":28,             # ResNet depth
    "width":2,              # ResNet width
    "large":false,          # flag of using large model(i.e., 135 filter size)
    "num_classes":10,       # Number of class, e.g., CIFAR-10 : 10
    "num_label":250,        # The number of available label [250, 1000, 4000]
    "batch_size":64,        # batch size
    "epochs":1024,          # epoch
    "save_epoch":10,        # interval of saving checkpoint
    "resume": false,        # resuming the training
    "ckpt": "latest.pth",   # checkpoint name 
    "verbose": false,       # If True, print training log on the console

    /* Training Configuration */
    "lr":0.002,              
    "lambda_u": 75,   
    "optim":"ADAM",         # type of optimizer [Adam, SGD]
    "alpha":0.75,           
    "T" : 0.5,              # fixed across all experiments, but you can adjust it
    "K" : 2,                # fixed across all experiments, but you can adjust it
    "ema_alpha":0.999,
    "seed":2114             # Different seed yields different result
}
```

 - `lambda_u` : A hyper-parameter weighting the contribution of the unlabeled examples to the training loss
 - `alpha`    : Hyperparameter for the Beta distribution used in MixU
 - `T`        : Temperature parameter for sharpening used in MixMatch
 - `K`        : Number of augmentations used when guessing labels in MixMatch
 - `seed`     : A number to initialize the random sampling. The results might be changed if you use different seed since it leads to different sampling strategy.

### Training Example

Training MixMatch on WideResNet28x2 using a CIFAR10 with 250 labeled data

> python main.py --cfg_path config/mixmatch/CIFAR10/train_CIFAR10_250.json

### Evaluation Example

Evaluating MixMatch on WideResNet28x2 using a CIFAR10 with 250 labeled data

> python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_250.json

## :gift: Pre-trained model

We provide the pre-trained model of CIFAR10 dataset. You can easily download the checkpoint files using below commands.
This shell file will automatically download the files and organize them to the desired path. The default result directory is `results`.
For those who cannot download the files using shell file, access the [link](https://drive.google.com/drive/folders/1Fjh-9aSvhAVYrxxXkxnrtW5s6yrprjRs?usp=sharing) directly.
In the case of downloading the file directly, plz modify the `"ckpt": $checkpoint_name` in the configuration file. For instance, `"ckpt": Mixmatch_250.pth`.

```
bash experiments/download.sh
python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_250.json
python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_500.json
python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_1000.json
python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_2000.json
python main.py --cfg_path config/mixmatch/CIFAR10/eval_CIFAR10_4000.json
```
## :link: Experiments

### Table

**CIFAR-10** | 250  | 500 | 1000 | 2000 | 4000 |
| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
#Paper | 88.92±0.87	| 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 | 
**Repo #Shallow** | 88.53 | 88.60 | 90.72 | 93.10 | 93.27 | 

**SVHN** | 250  | 500 | 1000 | 2000 | 4000 |
| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
#Paper | 96.22±0.87	| 96.36±0.94 | 96.73±0.32 | 96.96±0.15 | 97.11±0.06 | 
**Repo #Shallow** | 94.10 | 0 | 0 | 0 | 95.11  | 96.08 | 

### Training log

We provide a board to monitor log values.
Follow the below commands to view the progress.

```bash
cd results/${name}
tensorboard --logdir=log/ --bind_all
```

## Reference

- YU1ut [MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch)  
- perrying [realistic-ssl-evaluation-pytorch](https://github.com/perrying/realistic-ssl-evaluation-pytorch)  
- google-research [mixmatch](https://github.com/google-research/mixmatch)  


```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```