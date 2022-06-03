import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import spline

import pandas as pd


if __name__ == '__main__':
    show = False
    # show = True

    frac = 0.1
    iid = 0
    # lr = 0.01
    # batch_size = 10

    lr = 0.1
    batch_size = 400
    length = 3000
    seeds = [1, 5000, 6000, 9000]
    measure = "loss"
    # measure = "accuracy"
    # figure = "test_acc"
    # figure = "user_accu"
    figure = "improved"
    ####################################################################
    folder_path = ""
    if figure == "test_acc":
        name = folder_path + "final_test_accuracy_iid{}_frac{}_B{}_global_rate".format(iid, frac, batch_size)
        bias = -2
    else:
        if measure == "loss":
            name = folder_path + "final_percentage_loss_iid{}_frac{}_B{}_global_rate".format(iid, frac, batch_size)
            bias = 1
        elif measure == "accuracy":
            name = folder_path + "final_percentage_accu_iid{}_frac{}_B{}_global_rate".format(iid, frac, batch_size)
            bias = 0

    # label = ["FedAvg", r"FedMGDA-n, $\eta'$=1", r"FedMGDA-n, $\eta'$=1, decay", r"FedMGDA-n, $\eta'$=1.5, decay",
    #          r"FedMGDA-n, $\eta'$: adaptive"]
    # label = ["FedMGDA", r"FedMGDA-n, $\eta'$=1", r"FedMGDA-n, $\eta'$=1, decay", r"FedMGDA-n, $\eta'$=1.5, decay",
    #          r"FedMGDA-n, $\eta'$: adaptive"]
    # label = ["FedAvg", r"FedProx, $\mu=0.1$", "FedMGDA", r"FedMGDA+, $\eta'$=1",  r"FedAvg-n, $\eta'$=1", "q-FedAvg, q=0.5", r"FedMGDA-Prox, $\eta'$=1"]
    # label = ["FedAvg", r"FedProx", "FedMGDA", r"FedMGDA+",  r"FedAvg-n", "q-FedAvg", r"MGDA-Prox"]
    label = ["FedAvg", "FedProx", "FedMGDA", "FedMGDA+", "MGDA-Prox", "FedAvg-n", r"$q$-FedAvg"]
    # r"FedMGDA-n, $\eta'$=1, decay", r"FedMGDA-n, $\eta'$=1.5, decay",
    # r"FedMGDA-n, $\eta'$: adaptive"]
    ####################################################################

    result_res = np.zeros((9, 4, 4, length)) * np.nan
    result_accu = np.zeros((9, 4, 1, int(length / 10))) * np.nan
    user_accu = np.zeros((9, 4, int(length / 10), 100)) * np.nan
    user_loss = np.zeros((9, 4, int(length / 10), 100)) * np.nan

    folder_path = [""]

    res_name = [
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0, 1.0,
                                                                                                           0.0, 0.0,
                                                                                                           1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(
            0, 0, length, frac, iid, batch_size, lr, 1.0, 1.0, 0.1, 0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 1, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0, 1.0,
                                                                                                           0.0, 0.0,
                                                                                                           1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0,
                                                                                                           0.89, 0.0,
                                                                                                           0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0,
                                                                                                           0.89, 0.1,
                                                                                                           0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 0, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0,
                                                                                                           0.89, 0.0,
                                                                                                           0.0, 1.0),
        "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(0, 0, length,
                                                                                                           frac, iid,
                                                                                                           batch_size,
                                                                                                           lr, 1.0, 1.0,
                                                                                                           0.0, 0.5,
                                                                                                           1.0),

        # "N{}_EPS{}.0_cifar_cnn_epochs{}_C{}_iid{}_E1_B{}_lr{}_lrs{}_decay{}_prox{}_qffl{}_Lipsch{}".format(1, 1, length, frac, iid, batch_size, lr, 1.0, 0.926, 0.1, 0.0, 1.0),
    ]

    seed = [1, 5000, 6000, 9000]
    for idx in range(len(res_name)):
        for i, s in enumerate(seed):
            file = folder_path[0] + res_name[idx] + "_seed{}.npz".format(s)
            if os.path.isfile(file):
                temp = np.load(file)
                result_res[idx, i] = temp["res"]
                # result_accu[idx, i] = temp["acc"]
                # user_accu[idx, i] = temp["user"][0]
                # user_loss[idx, i] = temp["user"][1]
                print("hi{}".format(idx))

    if figure == "user_accu":
        x = np.arange(0, length / 10) * 10
        result = np.mean(user_accu, axis=3)
        smooth = 5
        ylable = 'Average test accuracy of users %'
        for idx in range(len(res_name)):
            y = np.nanmean(result[idx], axis=0)
            df = pd.DataFrame({'x': x, 'y': y})
            yy = df.y.ewm(span=smooth).mean()
            plt.plot(x, yy * 100, label=label[idx], linewidth=4)
    else:

        if figure == "test_acc":
            result = result_accu
            x = np.arange(0, length / 10) * 10
            smooth = 5
            ylable = 'Test accuracy %'
        else:
            result = result_res
            x = np.arange(0, length)
            smooth = 50
            ylable = 'Improved users %'
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for idx in range(len(res_name)):
            if idx != 2:
                y = np.nanmean(result[idx, :, 1 * 2 + bias, :], axis=0)
                df = pd.DataFrame({'x': x, 'y': y})
                yy = df.y.ewm(span=smooth).mean()
                plt.plot(x, yy * 100, label=label[idx], linewidth=4, color=colors[idx])

    plt.legend(fontsize=35, ncol=3)
    # plt.title(r'FedAvg with  $\alpha$-smoothed objective function', fontsize=45)
    plt.xlabel('Communication rounds', fontsize=45)
    plt.ylabel(ylable, fontsize=45)
    plt.ylim((41, 110))
    # plt.grid(which='both')
    plt.tick_params(labelsize=45)
    if show:
        plt.show()
    else:
        fig = plt.gcf()
        fig.set_size_inches(22, 22 / 2)
        name = name.replace(".", "")
        fig.savefig(name + ".eps", format='eps', dpi=600, bbox_inches='tight')
        fig.savefig(name + '.png', format='png', dpi=600, bbox_inches='tight')
        print('final')




