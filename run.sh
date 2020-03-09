#!/bin/bash

echo "running cnn on cifar from 0 to 1"
for i in $(seq 0 0.1 1) 
do
python3 federated_main.py "--epochs=2000" "--local_ep=1" "--model=cnn" "--normalize=0" "--epsilon=$i" "--dataset=cifar" "--iid=1" "--seed=1" "--gpu=cuda:0"
done
