#!/bin/bash

for i in 0 0.05 0.1 0.5 1
do
    for j in 1 2 3 4 5
	do
	    python3 federated_main.py --epochs=2000 --local_ep=1 --model=cnn --normalize=0 --epsilon=$i --dataset=cifar --iid=1 --seed=$j --gpu=cuda:0 --momentum=0 --inference=0
    done
done
