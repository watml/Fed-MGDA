#!/usr/bin/env python

import pickle
import matplotlib
import matplotlib.pyplot as plt


filename = 'N[0]EPS[0.01]cifar_cnn_2000_C[0.1]_iid[1]_E[1]_B[10].pickle'

infile = open(filename,'rb')
list_loss_accuracy = pickle.load(infile)
infile.close()

train_loss, train_accuracy = list_loss_accuracy


matplotlib.use('Agg')

plt.figure()
plt.title('Training Loss vs Communication rounds')
plt.plot(range(len(train_loss)), train_loss, color='r')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.savefig('N[0]EPS[0.01]cifar_cnn_2000_C[0.1]_iid[1]_E[1]_B[10]_loss.png')

# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average User Accuracy vs Communication rounds')
plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
plt.ylabel('Average User Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('N[0]EPS[0.01]cifar_cnn_2000_C[0.1]_iid[1]_E[1]_B[10]_acc.png')