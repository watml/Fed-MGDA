#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)



def ewm_average(olist): # smoothing the plot
    length = len(olist)
    x = np.arange(0,length)
    df = pd.DataFrame({'x': x, 'y': olist})
    avg_list = df.y.ewm(span=50).mean()
    return avg_list

if __name__ == '__main__':
    # epsilon=0.0
    filename = ["", "", "", "", ""]
    n = len(filename)
    for i in range(n):
        filename[i] = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
            2000,0.1, 1,10, 0.01, 1.0, 0.0, 'cnn',0 , 0.0, 'cifar', 1, i+1, 1.0, 0.0, 0.0, 1.0)

    # for i in range(n):
    #     filename[i] = "Ep[2000]_frac[0.1]_1_10_0.01_Mom[0.0]_cnn_N[0]_EPS[{}]_cifar_iid[1]_seed[{}].pickle".format(0.0,
    #                                                                                                                i + 1)

    train_loss = []
    train_accuracy = []

    for i in range(n):
        infile = open(filename[i], 'rb')
        loss, accuracy = pickle.load(infile)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        infile.close()

    avg_train_loss1 = []
    avg_train_accuracy1 = []
    for i in range(len(train_loss[0])):
        avg_loss = (train_loss[0][i] + train_loss[1][i] + train_loss[2][i] + train_loss[3][i] + train_loss[4][i]) / 5
        avg_train_loss1.append(avg_loss)

    for i in range(len(train_accuracy[0])):
        avg_accuracy = 100 * (
                    train_accuracy[0][i] + train_accuracy[1][i] + train_accuracy[2][i] + train_accuracy[3][i] +
                    train_accuracy[4][i]) / 5
        avg_train_accuracy1.append(avg_accuracy)

    # In[12]:

    # epsilon=0.1
    filename = ["", "", "", "", ""]
    n = len(filename)
    for i in range(n):
        filename[i] = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
            2000,0.1, 1,10, 0.01, 1.0, 0.0, 'cnn',0 , 0.0, 'cifar', 1, i+1, 1.0, 0.0, 0.0, 1.0)

    # for i in range(n):
    #     filename[i] = "Ep[2000]_frac[0.1]_1_10_0.01_Mom[0.0]_cnn_N[0]_EPS[{}]_cifar_iid[1]_seed[{}].pickle".format(0.1,
    #                                                                                                                i + 1)

    train_loss = []
    train_accuracy = []

    for i in range(n):
        infile = open(filename[i], 'rb')
        loss, accuracy = pickle.load(infile)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        infile.close()

    avg_train_loss2 = []
    avg_train_accuracy2 = []
    for i in range(len(train_loss[0])):
        avg_loss = (train_loss[0][i] + train_loss[1][i] + train_loss[2][i] + train_loss[3][i] + train_loss[4][i]) / 5
        avg_train_loss2.append(avg_loss)

    for i in range(len(train_accuracy[0])):
        avg_accuracy = 100 * (
                    train_accuracy[0][i] + train_accuracy[1][i] + train_accuracy[2][i] + train_accuracy[3][i] +
                    train_accuracy[4][i]) / 5
        avg_train_accuracy2.append(avg_accuracy)

    # In[13]:

    # epsilon=0.5
    filename = ["", "", "", "", ""]
    n = len(filename)
    for i in range(n):
        filename[i] = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
            2000,0.1, 1,10, 0.01, 1.0, 0.0, 'cnn',0 , 0.0, 'cifar', 1, i+1, 1.0, 0.0, 0.0, 1.0)

    # for i in range(n):
    #     filename[i] = "Ep[2000]_frac[0.1]_1_10_0.01_Mom[0.0]_cnn_N[0]_EPS[{}]_cifar_iid[1]_seed[{}].pickle".format(0.5,
    #                                                                                                                i + 1)

    train_loss = []
    train_accuracy = []

    for i in range(n):
        infile = open(filename[i], 'rb')
        loss, accuracy = pickle.load(infile)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        infile.close()

    avg_train_loss3 = []
    avg_train_accuracy3 = []
    for i in range(len(train_loss[0])):
        avg_loss = (train_loss[0][i] + train_loss[1][i] + train_loss[2][i] + train_loss[3][i] + train_loss[4][i]) / 5
        avg_train_loss3.append(avg_loss)

    for i in range(len(train_accuracy[0])):
        avg_accuracy = 100 * (
                    train_accuracy[0][i] + train_accuracy[1][i] + train_accuracy[2][i] + train_accuracy[3][i] +
                    train_accuracy[4][i]) / 5
        avg_train_accuracy3.append(avg_accuracy)

    # In[14]:

    # epsilon=1.0
    filename = ["", "", "", "", ""]
    n = len(filename)
    for i in range(n):
        filename[i] = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
            2000,0.1, 1,10, 0.01, 1.0, 0.0, 'cnn',0 , 0.0, 'cifar', 1, i+1, 1.0, 0.0, 0.0, 1.0)

    # for i in range(n):
    #     filename[i] = "Ep[2000]_frac[0.1]_1_10_0.01_Mom[0.0]_cnn_N[0]_EPS[{}]_cifar_iid[1]_seed[{}].pickle".format(1.0,
    #                                                                                                                i + 1)

    train_loss = []
    train_accuracy = []

    for i in range(n):
        infile = open(filename[i], 'rb')
        loss, accuracy = pickle.load(infile)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        infile.close()

    avg_train_loss4 = []
    avg_train_accuracy4 = []
    for i in range(len(train_loss[0])):
        avg_loss = (train_loss[0][i] + train_loss[1][i] + train_loss[2][i] + train_loss[3][i] + train_loss[4][i]) / 5
        avg_train_loss4.append(avg_loss)

    for i in range(len(train_accuracy[0])):
        avg_accuracy = 100 * (
                    train_accuracy[0][i] + train_accuracy[1][i] + train_accuracy[2][i] + train_accuracy[3][i] +
                    train_accuracy[4][i]) / 5
        avg_train_accuracy4.append(avg_accuracy)

    # In[15]:

    # epsilon=0.05
    filename = ["", "", "", "", ""]
    n = len(filename)
    for i in range(n):
        filename[i] = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
            2000,0.1, 1,10, 0.01, 1.0, 0.0, 'cnn',0 , 0.0, 'cifar', 1, i+1, 1.0, 0.0, 0.0, 1.0)

    # for i in range(n):
    #     filename[i] = "Ep[2000]_frac[0.1]_1_10_0.01_Mom[0.0]_cnn_N[0]_EPS[{}]_cifar_iid[1]_seed[{}].pickle".format(0.05,
    #                                                                                                                i + 1)

    train_loss = []
    train_accuracy = []

    for i in range(n):
        infile = open(filename[i], 'rb')
        loss, accuracy = pickle.load(infile)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        infile.close()

    avg_train_loss5 = []
    avg_train_accuracy5 = []
    for i in range(len(train_loss[0])):
        avg_loss = (train_loss[0][i] + train_loss[1][i] + train_loss[2][i] + train_loss[3][i] + train_loss[4][i]) / 5
        avg_train_loss5.append(avg_loss)

    for i in range(len(train_accuracy[0])):
        avg_accuracy = 100 * (
                    train_accuracy[0][i] + train_accuracy[1][i] + train_accuracy[2][i] + train_accuracy[3][i] +
                    train_accuracy[4][i]) / 5
        avg_train_accuracy5.append(avg_accuracy)

    # In[16]:

    avg_train_loss1 = ewm_average(avg_train_loss1)
    avg_train_loss2 = ewm_average(avg_train_loss2)
    avg_train_loss3 = ewm_average(avg_train_loss3)
    avg_train_loss4 = ewm_average(avg_train_loss4)
    avg_train_loss5 = ewm_average(avg_train_loss5)

    avg_train_accuracy1 = ewm_average(avg_train_accuracy1)
    avg_train_accuracy2 = ewm_average(avg_train_accuracy2)
    avg_train_accuracy3 = ewm_average(avg_train_accuracy3)
    avg_train_accuracy4 = ewm_average(avg_train_accuracy4)
    avg_train_accuracy5 = ewm_average(avg_train_accuracy5)

    # In[17]:

    fig1, ax1 = plt.subplots()

    ax1.tick_params(labelsize=45)

    plt.title('Training Loss vs Communication rounds', fontsize=45)
    ax1.plot(range(len(avg_train_loss1[20:])), avg_train_loss1[20:], linewidth=4)
    ax1.plot(range(len(avg_train_loss5[20:])), avg_train_loss5[20:], linewidth=4)
    ax1.plot(range(len(avg_train_loss2[20:])), avg_train_loss2[20:], linewidth=4)
    ax1.plot(range(len(avg_train_loss3[20:])), avg_train_loss3[20:], linewidth=4)
    ax1.plot(range(len(avg_train_loss4[20:])), avg_train_loss4[20:], linewidth=4)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_ticks(np.arange(0, 2.25, 0.25))

    ax1.legend(['$\epsilon$=0', '$\epsilon$=0.05', '$\epsilon$=0.1', '$\epsilon$=0.5', '$\epsilon$=1.0'],
               loc='upper right', fontsize=35, ncol=5, mode="expand")

    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.18, 0.4, 0.5, 0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

    ax1.set_ylabel('Training loss', fontsize=45)
    ax1.set_xlabel('(d)', fontsize=55)

    ax2.set_xlim(980, 2005)
    ax2.set_ylim(-0.02, 0.51)

    ax2.plot(range(len(avg_train_loss1)), avg_train_loss1, linewidth=5)
    ax2.plot(range(len(avg_train_loss5)), avg_train_loss5, linewidth=5)
    ax2.plot(range(len(avg_train_loss2)), avg_train_loss2, linewidth=5)
    ax2.plot(range(len(avg_train_loss3)), avg_train_loss3, linewidth=5)
    ax2.plot(range(len(avg_train_loss4)), avg_train_loss4, linewidth=5)
    ax2.tick_params(labelsize=0)

    fig1 = plt.gcf()
    fig1.set_size_inches(22, 22 / 1.28)
    fig1.savefig('figure1d' + '.eps', format='eps', dpi=600, bbox_inches='tight')
    # iid_interpolation_loss_thick_nogrid

    # Plot Average Accuracy vs Communication rounds
    fig2, ax1 = plt.subplots()

    ax1.tick_params(labelsize=45)

    plt.title('Average User Accuracy vs Communication rounds', fontsize=45)
    ax1.plot(range(len(avg_train_accuracy1)), avg_train_accuracy1, linewidth=3.5)
    ax1.plot(range(len(avg_train_accuracy5)), avg_train_accuracy5, linewidth=3.5)
    ax1.plot(range(len(avg_train_accuracy2)), avg_train_accuracy2, linewidth=3.5)
    ax1.plot(range(len(avg_train_accuracy3)), avg_train_accuracy3, linewidth=3.5)
    ax1.plot(range(len(avg_train_accuracy4)), avg_train_accuracy4, linewidth=3.5)

    # ax1.legend(['$\epsilon$=0','$\epsilon$=0.05', '$\epsilon$=0.1','$\epsilon$=0.5','$\epsilon$=1.0'], loc='upper right',fontsize=35,ncol=5, mode="expand")

    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.15, 0.1, 0.5, 0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

    ax1.set_ylabel('Avg user accuracy %', fontsize=45)
    ax1.set_xlabel('(b)', fontsize=55)

    ax2.set_xlim(990, 2005)
    ax2.set_ylim(70.75, 72.25)

    ax2.plot(range(len(avg_train_accuracy1)), avg_train_accuracy1, linewidth=5)
    ax2.plot(range(len(avg_train_accuracy5)), avg_train_accuracy5, linewidth=5)
    ax2.plot(range(len(avg_train_accuracy2)), avg_train_accuracy2, linewidth=5)
    ax2.plot(range(len(avg_train_accuracy3)), avg_train_accuracy3, linewidth=5)
    ax2.plot(range(len(avg_train_accuracy4)), avg_train_accuracy4, linewidth=5)
    ax2.tick_params(labelsize=0)

    fig2 = plt.gcf()
    fig2.set_size_inches(22, 22 / 1.32)
    fig2.savefig('figure1b' + '.eps', format='eps', dpi=600, bbox_inches='tight')
    # iid_interpolation_acc_thick_nogrid

    # In[ ]:






