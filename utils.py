#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid, mnist_noniid_unequal
from sampling import noniid_adult
from torch.utils import data
import numpy as np
import quadprog


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'adult':
        print('args.dataset', args.dataset)
        data_dir = './data/adult/numpy/'
        train_X = np.load(data_dir + 'X_train.npy')
        train_y = np.load(data_dir + 'y_train.npy')
        test_X = np.load(data_dir + 'X_test.npy')
        test_y = np.load(data_dir + 'y_test.npy')
        train_dataset = data.TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_y))
        test_dataset = data.TensorDataset(torch.from_numpy(test_X),torch.from_numpy(test_y))
        if args.iid:
            user_groups = iid(train_dataset, args.num_users)
        else:
            print('noniid')
            user_groups = noniid_adult(train_dataset, args.num_users)  

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, args.num_users, 500, 100)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        elif args.dataset == 'fmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)


        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, args.num_users, 300, 200)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

# def solve_w(U):
#     n,d = U.shape
#     K = np.eye(n,dtype=float)
#     for i in range(0,n):
#         for j in range(0,n):
#             K[i,j]= np.dot(U[i],U[j])
#     Q = 0.5 *(K + K.T)
#     p = np.zeros(n,dtype=float)
#     a = np.ones(n,dtype=float).reshape(-1,1)
#     Id = np.eye(n,dtype=float)
#     A = np.concatenate((a,Id),axis=1)
#     b = np.zeros(n+1)
#     b[0] = 1.
#     return U.T.dot(quadprog.solve_qp(Q,p,A,b)[0])


def solve_centered_w(U,epsilon):
    """
        U is a list of gradients (stored as state_dict()) from n users
    """

    n = len(U)
    K = np.eye(n,dtype=float)
    for i in range(0,n):
        for j in range(0,n):
            K[i,j] = 0
            for key in U[i].keys():
                K[i,j] += torch.mul(U[i][key],U[j][key]).sum()

    Q = 0.5 *(K + K.T)
    p = np.zeros(n,dtype=float)
    a = np.ones(n,dtype=float).reshape(-1,1)
    Id = np.eye(n,dtype=float)
    neg_Id = -1. * np.eye(n,dtype=float)
    lower_b = (1./n - epsilon) * np.ones(n,dtype=float)
    upper_b = (-1./n - epsilon) * np.ones(n,dtype=float)
    A = np.concatenate((a,Id,Id,neg_Id),axis=1)
    b = np.zeros(n+1)
    b[0] = 1.
    b_concat = np.concatenate((b,lower_b,upper_b))
    alpha = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
    print(alpha)
    return alpha


def solve_capped_w(U,C=1):
    """
        U is a list of gradients (stored as state_dict()) from n users
    """

    n = len(U)
    K = np.eye(n,dtype=float)
    for i in range(0,n):
        for j in range(0,n):
            K[i,j] = 0
            for key in U[i].keys():
                K[i,j] += torch.mul(U[i][key],U[j][key]).sum()

    Q = 0.5 *(K + K.T)
    p = np.zeros(n,dtype=float)
    a = np.ones(n,dtype=float).reshape(-1,1)
    Id = np.eye(n,dtype=float)
    neg_Id = -1. * np.eye(n,dtype=float)
    cap_b = (-C) * np.ones(n,dtype=float)
    A = np.concatenate((a,Id,neg_Id),axis=1)
    b = np.zeros(n+1)
    b[0] = 1.
    b_concat = np.concatenate((b,cap_b))
    alpha = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
    print(alpha)
    return alpha