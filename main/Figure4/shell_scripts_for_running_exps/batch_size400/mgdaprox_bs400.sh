#!/bin/bash

for j in 1 5000 6000 9000
do
    python3 federated_main.py --seed=$j --normalize=1 --epsilon=1 --epochs=3000 --local_bs=400 --lr=0.1 --global_lr_decay=0.89 --prox_weight=0.1 --qffl=0 --Lipschitz_constant=1 --iid=0 --gpu=cuda:0 --momentum=0 --local_ep=1 --inference=1
done
