# Multi-Objective Federated-Learning (PyTorch)
Codebase and original Readme adopted from: https://github.com/AshwinRJ/Federated-Learning-PyTorch 


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install the following packages 
* Python3
* Pytorch
* Torchvision
* Numpy
* tensorboardX
* tqdm
* quadprog
* pickle

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN with FedAvg (IID):
```
python3 src/federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=0 --dataset=cifar --gpu=cuda:0
```
* To run the federated experiment with CIFAR on CNN with FedMGDA (non-IID):
```
python3 src/federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=1 --dataset=cifar --gpu=cuda:0 --iid=0
```
* To run the federated experiment with CIFAR on CNN with normalized FedMGDA with vanilla SGD and record more training information (IID):
```
python3 src/federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=1 --dataset=cifar --gpu=cuda:0 --normalize=1 --momentum=0 --inference=1
```
* To run the federated experiment with CIFAR on CNN with capped FedMGDA (non-IID):
```
python3 src/federated_main.py --epochs=2000 --local_ep=1 --model=cnn --epsilon=1 --cap=0.5 --dataset=cifar --gpu=cuda:0 --normalize=1 --momentum=0 --iid=0
```


You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

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

## Results on MNIST
#### Baseline Experiment:
To be updated

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

To be updated

## Further Readings
### Papers:
* [Steepest descent methods for multicriteria optimization](https://link.springer.com/article/10.1007/s001860000043)
* [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146)
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)