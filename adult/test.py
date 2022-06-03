import torch
from models import Adult
import numpy as np
from torch import nn
from sampling import iid
from sklearn.linear_model import LogisticRegression

adult = Adult()
df = '../data/adult/numpy/'
X_train = np.load(df + 'X_train.npy')
X_test = np.load(df + 'X_test.npy')
y_train = np.load(df + 'y_train.npy')
y_test = np.load(df + 'y_test.npy')
X_train_phd = np.load(df + 'X_train_phd.npy')
y_train_phd = np.load(df + 'y_train_phd.npy')
X_train_non_phd = np.load(df + 'X_train_non_phd.npy')
y_train_non_phd = np.load(df + 'y_train_non_phd.npy')
X_test_phd = np.load(df + 'X_test_phd.npy')
y_test_phd = np.load(df + 'y_test_phd.npy')
X_test_non_phd = np.load(df + 'X_test_non_phd.npy')
y_test_non_phd = np.load(df + 'y_test_non_phd.npy')
X_train_phd = torch.from_numpy(X_train_phd)
sample = X_train_phd[0].float()

if __name__ == '__main__':
    print(len(X_train))
    #print(iid(X_train_phd, 100))
    print(adult(sample))
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    print("predict: ", clf.predict(X_test[:2, :]))
    print("score: ", clf.score(X_train, y_train))
    print("score: ", clf.score(X_test, y_test))

