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

from torch.utils import data
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, Adult
from utils import get_dataset, average_weights, exp_details,solve_centered_w,project_to_simplex
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

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### to ensure better reproducibility, following must be set
    ### However, these may slow down the processing speed

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # On Adult dataset, we also test performance on phd and non-phd domain
    if args.dataset == 'adult':
        data_dir = './data/adult/numpy/'
        X_test_phd = np.load(data_dir + 'X_test_phd.npy')
        y_test_phd = np.load(data_dir + 'y_test_phd.npy')
        X_test_non_phd = np.load(data_dir + 'X_test_non_phd.npy')
        y_test_non_phd = np.load(data_dir + 'y_test_non_phd.npy')
        phd_test_dataset = data.TensorDataset(torch.from_numpy(X_test_phd),torch.from_numpy(y_test_phd))
        non_phd_test_dataset = data.TensorDataset(torch.from_numpy(X_test_non_phd), torch.from_numpy(y_test_non_phd))


    # BUILD MODEL

    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            print('using Fashion MNIST')
            #global_model = CNNFashion_Mnist(args=args)
            global_model = CNNMnist(args=args)
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
    elif args.model == 'lr':  # logistic regression
        if args.dataset == 'adult':
            global_model = Adult()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    model_total_params = [p.numel() for p in global_model.parameters() if p.requires_grad]
    print('model trainable parameter count: sum{}={}\n'.format(model_total_params, sum(model_total_params)))

    # copy weights
    old_global_weights = global_model.state_dict()

    # Training
    if args.inference:
        train_loss_improve_percentage = []
        train_acc_improve_percentage = []
        # followings are new variables
        train_loss_improve_percentage_post = []
        train_acc_improve_percentage_post = []
        test_acc_users = []
    train_loss, train_accuracy = [], []
    user_test_accuracy, user_test_loss = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 10
    val_loss_pre, counter = 0, 0
    # new
    total_accuracy = []
    vip_training_loss, vip_training_accuracy = [], []

    eta = args.global_lr
    decay = args.global_lr_decay

    latest_lambdas = [0.5,0.5]
    list_latest_lambdas = []
    list_models = []

    for epoch in tqdm(range(args.epochs)):
        # local_weights, local_losses = [], []
        local_weights, local_losses, local_norms = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        print('m is: ',m)


        idxs_users = [0,1]
        # if args.vip == -1:
        #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #     print('idxs_users is:',idxs_users)
        # else:
        #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #     if args.vip in idxs_users:
        #         idx_vip = np.where(idxs_users==args.vip)
        #         idxs_users[idx_vip], idxs_users[-1] = idxs_users[-1], idxs_users[idx_vip]
        #     else:
        #         idxs_users[-1] = args.vip


        # Do an inference on selected participants to get training loss, training accuracy, testing accuracy
        if args.inference:
            # user_train_accuracy_list, user_train_loss_list = [], []
            user_train_accuracy_list, user_train_loss_list, user_train_accu_post, user_train_loss_post = [], [], [], []
            for idx in idxs_users:
                local_model = LocalUpdate(args=args,dataset=train_dataset,idxs=user_groups[idx],logger=logger)
                user_train_accuracy, user_train_loss = local_model.inference_trainloader(model=global_model)
                user_train_accuracy_list.append(user_train_accuracy)
                user_train_loss_list.append(user_train_loss)
        else:
            user_train_accuracy_list, user_train_loss_list, user_train_accu_post, user_train_loss_post = [], [], [], []
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                user_train_accuracy, user_train_loss = local_model.inference_trainloader(model=global_model)
                user_train_accuracy_list.append(user_train_accuracy)
                user_train_loss_list.append(user_train_loss)



        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            #w, loss = local_model.update_weights(
            #    model=copy.deepcopy(global_model), global_round=epoch)
            w, loss, loss_post, accuracy_post, norm = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            # new
            local_norms.append(copy.deepcopy(norm))
            user_train_accu_post.append(copy.deepcopy(accuracy_post))
            user_train_loss_post.append(copy.deepcopy(loss_post))


        if args.afl: # Update the global model using AFL
            delta_w = copy.deepcopy(old_global_weights)
            for key in delta_w.keys():
                delta_w[key] = latest_lambdas[0]*local_weights[0][key]+ latest_lambdas[1]*local_weights[1][key]
            new_global_weights = copy.deepcopy(old_global_weights)
            for key in new_global_weights.keys():
                new_global_weights[key] += delta_w[key] * args.lr_afl_weight

            for idx in range(len(latest_lambdas)):
                latest_lambdas[idx] += args.lr_lambda * local_losses[idx]
            latest_lambdas = project_to_simplex(latest_lambdas)

            list_models.append(new_global_weights)
            list_latest_lambdas.append(latest_lambdas)

        else:
            if args.qffl == 0:  # Update the global model using  {FedAvg, FedMGDA, FedProx, FedMGDAProx}
                n = len(local_weights)
                if args.epsilon <= 1 and args.epsilon != 0:
                    gradient_coefficients = solve_centered_w(local_weights, epsilon=args.epsilon)
                elif args.epsilon == 0:
                    gradient_coefficients = np.array([0.5, 0.5], dtype=float)

                # update global weights
                new_global_weights = copy.deepcopy(old_global_weights)

                if decay <= 1:
                    lr_server = eta * (decay ** (epoch // 100))
                elif decay == 2:
                    temp = np.array(local_norms)
                    lr_server = np.median(temp)

                for key in new_global_weights.keys():
                    for i in range(n):
                        new_global_weights[key] += torch.mul(local_weights[i][key],
                                                             gradient_coefficients[i] * lr_server)
            else:  # Update the global model using q-FedAvg
                new_global_weights = copy.deepcopy(old_global_weights)
                dw = copy.deepcopy(local_weights)
                for i in range(len(dw)):
                    for key in dw[i].keys():
                        dw[i][key] *= args.Lipschitz_constant
                dw_norm = [0] * len(dw)
                for i in range(len(dw_norm)):
                    total = 0
                    for key in dw[i].keys():
                        total += torch.norm(dw[i][key]) ** 2
                    dw_norm[i] = total
                delta = copy.deepcopy(dw)
                for i in range(len(delta)):
                    for key in delta[i].keys():
                        # manipulate the phd loss, add bias to it, here the default bias is 0.0
                        if i == 0:
                            delta[i][key] *= (user_train_loss_list[i] + 0.0) ** args.qffl
                            # print('manipulated loss is:', (user_train_loss_list[i] + 0.0))
                        else:
                            delta[i][key] *= user_train_loss_list[i] ** args.qffl
                h = [0] * len(delta)
                for i in range(len(h)):
                    if i == 0:
                        h[i] = args.qffl * (user_train_loss_list[i] + 0.0) ** (args.qffl - 1) * dw_norm[i] \
                               + args.Lipschitz_constant * (user_train_loss_list[i] + 0.0) ** args.qffl
                        # print('manipulated h is: ', h[i])
                    else:
                        h[i] = args.qffl * user_train_loss_list[i] ** (args.qffl - 1) * dw_norm[i] \
                               + args.Lipschitz_constant * user_train_loss_list[i] ** args.qffl
                        # print('unmanipulated h is: ', h[i])
                s = sum(h)
                for key in new_global_weights.keys():
                    for i in range(len(delta)):
                        new_global_weights[key] += torch.mul(delta[i][key], 1 / s)




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
            if args.vip != -1:
                vip_training_accuracy.append(user_train_accuracy)
                vip_training_loss.append(user_train_loss)

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

            # NEW. Compare the fraction of improvements pre vs post training
            train_loss_improve_post = 0.0
            train_acc_improve_post = 0.0
            for i in range(len(user_train_loss_list)):
                if user_train_loss_post[i] <= user_train_loss_list[i]:
                    train_loss_improve_post += 1
                if user_train_accu_post[i] >= user_train_accuracy_list[i]:
                    train_acc_improve_post += 1
            train_loss_improve_post /= len(user_train_loss_post)
            train_acc_improve_post /= len(user_train_accu_post)

            train_loss_improve_percentage_post.append(train_loss_improve_post)
            train_acc_improve_percentage_post.append(train_acc_improve_post)

            print("percentage of improved participants (train_loss) is: {}".format(train_loss_improve))
            print("percentage of improved participants (train_acc) is: {}".format(train_acc_improve))
            print("percentage of improved participants (train_loss pre vs post) is: {}".format(train_loss_improve_post))
            print("percentage of improved participants (train_acc pre vs post) is: {}".format(train_acc_improve_post))

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        if epoch % print_every == 0:
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

        # new
        if epoch % print_every == 0:
            user_test_accuracy.append(list_acc)
            user_test_loss.append(list_loss)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # new. Test inference on test set every once a while
        if epoch % (print_every) == 0:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            total_accuracy.append(test_acc)


    final_model = average_weights(list_models)
    global_model.load_state_dict(final_model)


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)


    # For Adult dataset, also test on phd and non-phd domain
    if args.dataset == 'adult':
        phd_test_acc, phd_test_loss = test_inference(args,global_model,phd_test_dataset)
        non_phd_test_acc, non_phd_test_loss = test_inference(args, global_model, non_phd_test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    if args.dataset == 'adult':
        print("|---- Test Accuracy phd: {:.2f}%".format(100 * phd_test_acc))
        print("|---- Test Accuracy non-phd: {:.2f}%".format(100 * non_phd_test_acc))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Saving the objects train_loss and train_accuracy:

    # file_name = "save/objects/N[{}]EPS[{}]{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pickle".format(args.normalize,args.epsilon,args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    file_name = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].npz".format(
        args.epochs,
        args.frac, args.local_ep,
        args.local_bs, args.lr, args.global_lr,
        args.momentum, args.model,
        args.normalize, args.epsilon,
        args.dataset, args.iid, args.seed, args.global_lr_decay,args.prox_weight, args.qffl, args.Lipschitz_constant)

    file_name2 = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_{}_{}.log".format(
        args.epochs,
        args.frac, args.local_ep,
        args.local_bs, args.lr, args.global_lr,
        args.momentum, args.model,
        args.normalize, args.epsilon,
        args.dataset, args.iid, args.seed, args.global_lr_decay,args.prox_weight,args.lr_lambda,args.lr_afl_weight)

    file_name3 = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_{}_{}.pickle".format(
        args.epochs,
        args.frac, args.local_ep,
        args.local_bs, args.lr, args.global_lr,
        args.momentum, args.model,
        args.normalize, args.epsilon,
        args.dataset, args.iid, args.seed, args.global_lr_decay, args.prox_weight,args.lr_lambda,args.lr_afl_weight)

    if args.afl:
        folder = "save/objects/afl"
    elif args.qffl > 0:
        folder = "save/objects/qffl"
    else:
        folder = "save/objects/others"

    # else:
    #     if args.iid:
    #         folder = "save/objects/single_loss"
    #     else:
    #         folder = "save/objects/single_loss/nonphd"

    # if args.iid:
    #     folder = "save/objects/iid"
    # else:
    #     folder = "save/objects/noniid"

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_folder = os.path.join(THIS_FOLDER, folder)

    # make sure the folder exists
    if not os.path.isdir(my_folder):
        os.mkdir(my_folder)

    my_file = os.path.join(my_folder, file_name)
    my_file2 = os.path.join(my_folder, file_name2)
    my_file3 = os.path.join(my_folder, file_name3)

    if args.inference:
        np.savez(my_file, acc=[total_accuracy],
                 res=[train_acc_improve_percentage_post, train_loss_improve_percentage_post,
                      train_acc_improve_percentage, train_loss_improve_percentage],
                 vip=[vip_training_accuracy, vip_training_loss], user=[user_test_accuracy, user_test_loss])

    with open(my_file3, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    with open(my_file2,'w') as txtfile:
        txtfile.write("Configurations: AFL algorithm")
        txtfile.write("\n Learning rate for lambda: {}".format(args.lr_lambda))
        txtfile.write("\n Learning rate for weights: {}".format(args.lr_afl_weight))
        txtfile.write("\n Inference during training: {}".format(args.inference))
        txtfile.write("\n Global rounds: {}".format(args.epochs))
        txtfile.write("\n Number of users: {}".format(args.num_users))
        txtfile.write("\n Fraction of participants in each round: {}".format(args.frac))
        txtfile.write("\n Number of local epochs per round: {}".format(args.local_ep))
        txtfile.write("\n Proximal weight: {}".format(args.prox_weight))
        txtfile.write("\n Local minibatch size: {}".format(args.local_bs))
        txtfile.write("\n Local learning rate: {}".format(args.lr))
        txtfile.write("\n Global learning rate: {}".format(args.global_lr))
        txtfile.write("\n Global learning rate decay: {}".format(args.global_lr_decay))
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
        if args.dataset == 'adult':
            txtfile.write("\n |---- Test Accuracy phd: {:.2f}%".format(100 * phd_test_acc))
            txtfile.write("\n |---- Test Accuracy non-phd: {:.2f}%".format(100 * non_phd_test_acc))
        txtfile.write('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
        txtfile.write('\n Manipulate phd loss with bias 0, training loss is averaged')


