#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import quadprog
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details,solve_centered_w
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters





if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    old_global_weights = global_model.state_dict()

    # Training
    if args.inference:
        train_loss_improve_percentage = []
        train_acc_improve_percentage = []
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 10
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)



        # Do an inference on selected participants to get training loss, training accuracy, testing accuracy
        if args.inference:
            user_train_accuracy_list, user_train_loss_list = [], []
            for idx in idxs_users:
                local_model = LocalUpdate(args=args,dataset=train_dataset,idxs=user_groups[idx],logger=logger)
                user_train_accuracy, user_train_loss = local_model.inference_trainloader(model=global_model)
                user_train_accuracy_list.append(user_train_accuracy)
                user_train_loss_list.append(user_train_loss)


        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # compute common gradient direction
        n = len(local_weights)
        if args.epsilon != 0:
            gradient_coefficients = solve_centered_w(local_weights,epsilon=args.epsilon)
        else:
            gradient_coefficients = 1./n * np.ones(n,dtype=float)




        # update global weights
        new_global_weights = copy.deepcopy(old_global_weights)
        for key in new_global_weights.keys():
            for i in range(n):
                new_global_weights[key] += gradient_coefficients[i] * local_weights[i][key]



        # update global model
        global_model.load_state_dict(new_global_weights)

        # Do an inference on selected participants to get UPDATED training loss, training accuracy, testing accuracy
        if args.inference:
            new_user_train_accuracy_list, new_user_train_loss_list = [], []
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                user_train_accuracy, user_train_loss = local_model.inference_trainloader(model=global_model)
                new_user_train_accuracy_list.append(user_train_accuracy)
                new_user_train_loss_list.append(user_train_loss)

            train_loss_improve = 0.0
            train_acc_improve = 0.0

            # Compare the fraction of improvements
            for i in range(len(user_train_loss_list)):
                if new_user_train_loss_list[i] <= user_train_loss_list[i]:
                    train_loss_improve += 1
                if new_user_train_accuracy_list[i] >= user_train_accuracy_list[i]:
                    train_acc_improve += 1
            train_loss_improve /= len(user_train_loss_list)
            train_acc_improve /= len(user_train_accuracy_list)

            train_loss_improve_percentage.append(train_loss_improve)
            train_acc_improve_percentage.append(train_acc_improve)

            print("percentage of improved participants (train_loss) is: {}".format(train_loss_improve))
            print("percentage of improved participants (train_acc) is: {}".format(train_acc_improve))

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Saving the objects train_loss and train_accuracy:

    # file_name = "save/objects/N[{}]EPS[{}]{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pickle".format(args.normalize,args.epsilon,args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    
    file_name = "save/objects/Ep[{}]_frac[{}]_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}].pickle".format(args.epochs,
                                                                                            args.frac, args.local_ep,
                                                                                            args.local_bs, args.lr,
                                                                                            args.momentum, args.model,
                                                                                            args.normalize,args.epsilon,
                                                                                            args.dataset,args.iid,args.seed)

    file_name2 = "save/objects/Ep[{}]_frac[{}]_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}].log".format(
        args.epochs,
        args.frac, args.local_ep,
        args.local_bs, args.lr,
        args.momentum, args.model,
        args.normalize, args.epsilon,
        args.dataset, args.iid, args.seed)
    
    folder="save/objects"

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, file_name)
    my_file2 = os.path.join(THIS_FOLDER, file_name2)


    my_folder = os.path.join(THIS_FOLDER,folder)

    # make sure the folder exists
    if not os.path.isdir(my_folder):
        os.mkdir(my_folder)


    if args.inference:
        with open(my_file, 'wb') as f:
            pickle.dump([train_loss, train_accuracy,train_loss_improve_percentage,train_acc_improve_percentage], f)
    else:
        with open(my_file, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

    with open(my_file2,'w') as txtfile:
        txtfile.write("Configurations: ")
        txtfile.write("\n Inference during training: {}".format(args.inference))
        txtfile.write("\n Global rounds: {}".format(args.epochs))
        txtfile.write("\n Number of users: {}".format(args.num_users))
        txtfile.write("\n Fraction of participants in each round: {}".format(args.frac))
        txtfile.write("\n Number of local epochs per round: {}".format(args.local_ep))
        txtfile.write("\n Local minibatch size: {}".format(args.local_bs))
        txtfile.write("\n Local learning rate: {}".format(args.lr))
        txtfile.write("\n SGD momentum: {}".format(args.momentum))
        txtfile.write("\n NN model: {}".format(args.model))
        txtfile.write("\n Gradient Normalization: {}".format(args.normalize))
        txtfile.write("\n Centered-constraint epsilon: {}".format(args.epsilon))
        txtfile.write("\n Dataset: {}".format(args.dataset))
        txtfile.write("\n GPU: {}".format(args.gpu))
        txtfile.write("\n Optimizer used: {}".format(args.optimizer))
        txtfile.write("\n User data is split i.i.d : {}".format(args.iid))
        txtfile.write("\n In non-iid setting, data is split unequally: {}".format(args.unequal))
        txtfile.write("\n Random seed used: {}".format(args.seed))

        txtfile.write(' \n Results after {} global rounds of training:'.format(args.epochs))
        txtfile.write("\n |---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        txtfile.write("\n |---- Test Accuracy: {:.2f}%".format(100*test_acc))
        txtfile.write('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))




    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

