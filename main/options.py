#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inference', type=int, default=0,
                        help="Do inference during training for recording more advance information")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='local learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--global_lr', type=float, default=1.0,
                        help='global learning rate')
    parser.add_argument('--global_lr_decay', type=float, default=1.0,
                        help='Global learning rate decay every 100 rounds. Default is 1, means no decay.')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    # MGDA arguments
    parser.add_argument('--normalize', type=int, default=0,
                        help= "Default set to no normalization. Set to 1 for normalization")
    parser.add_argument('--epsilon', type=float, default=1.,
                        help = "Interpolation between FedMGDA and FedAvg. \
                        When set to 0, recovers FedAvg; When set to 1, is FedMGDA without any constraint")
    parser.add_argument('--cap', type=float, default=1.,
                        help="Capped MGDA parameter, when set to 1, same as default MGDA. \
                            Set to smaller values to restrict individual participation.")

    parser.add_argument('--vip', type=int, default=-1,
                        help='the ID of a user that participates in each communication round; {-1 no vip, 0....number of users}')

    # Proximal arguments
    parser.add_argument('--prox_weight', type=float, default=0.0,
                        help='the weight of proximal regularization term in FedProx and FedMGDA')

    # Q-fair federated learning
    parser.add_argument("--qffl", type=float, default=0.0, help="the q-value in the qffl algorithm. \
                                                                    qffl with q=0 reduces to FedAvg")
    parser.add_argument('--Lipschitz_constant', type=float, default=1.0)



    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
