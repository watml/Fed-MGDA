# FedMGDA
This is an repository of main code and experiments for the paper *Federated Learning Meets Multi-objective Optimization*

Experiments can be produced on MNIST, Fashion MNIST, CIFAR10 and Adult (both IID and non-IID). 

Since the purpose of these experiments are to illustrate and compare the effectiveness of different federated learning algorithms (including ours), only simple models such as MLP and CNN are used.

## Requirments
Install the following packages 
* Python  3.6.9 (Anaconda)
* PyTorch  1.3.0
* torchvision  0.4.1
* Numpy  1.17.2
* tensorboardX
* tqdm
* quadprog
* pickle

WARNING: we strongly recommend the reader to use the **same** version of Pytorch and torchvision as stated above, otherwise errors may appear.

## Quick start
For users who only want to validate or get a taste of the experiments mentioned in the paper (e.g. Figure 1) without spending too much effort, we provided off-the-shelf bash scripts and instructions in the corresponding folder (e.g. '/main/Figure1/shell_scripts_for_running_exps') which is easy to run. Scripts for reading the experiment output files and generating corresponding plots is also provided (e.g. '/main/Figure1/code_for_plotting').

For enthusiastic users who want to run the experiments flexibly, see the following sections.

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments on Mnist, Fashion Mnist and Cifar don't need manual download.
* Adult dataset is provided in this repo, no need to manually download or preprocess it.
* To use other datasets: move the dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments flexibly

* To run the federated experiment with CIFAR on CNN with FedAvg (IID):
```
python3 federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=0 --dataset=cifar --gpu=cuda:0
```
* To run the federated experiment with CIFAR on CNN with FedMGDA (non-IID):
```
python3 federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=1 --dataset=cifar --gpu=cuda:0 --iid=0
```
* To run the federated experiment on Fashion-MNIST on CNN with FedMGDA+ and record more training information (IID):
```
python3 federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=1 --dataset=fmnist --gpu=cuda:0 --normalize=1 --momentum=0 --inference=1
```
* To run the federated experiment on Adult with linear model with Agnostic Federated Learning (non-IID):
```
python3 federated_main.py --dataset=adult --model=lr --epochs=500 --iid=0 --local_ep=1 --normalize=0 --epsilon=0 --momentum=0 --frac=1 --local_bs=10 --inference=0 --num_users=2 --afl=1 --lr_lambda=0.1 --lr_afl_weight=1 --lr=0.01 --seed=1
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given for some of the parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--inference:```    Record more training information by doing inference during training. Default set to 0. Set to 1 to turn on. 
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.
* ```--momentum:```     Momentum applied in SGD optimizer. Default is 0.5

#### MGDA parameters
* ```--normalize:```      Apply normalization to gradients. Default set to no normalization. Set to 1 for normalization.
* ```--epsilon:```      Epsilon-centered constraints. Can be viewed as interpolation between FedMGDA and FedAvg. When set to 0, recovers FedAvg; When set to 1, is FedMGDA. 
* ```--cap:```      Capped MGDA parameter. When set to 1, same as default MGDA. Set to smaller values to restrict individual dominance.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.
