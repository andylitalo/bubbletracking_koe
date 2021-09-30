"""
train.py trains several machine learning models to distinguish objects from 
noise/artifacts based on the labeling performed in `label.py`.

I will try the following models:
    - Linear classifier
    - Logistic regression
    - SVM
    - Random Forest
    - Boosting (AdaBoost)
    - Neural Network (PyTorch)

Author: Andy Ylitalo
Date: September 29, 2021
"""

# 3rd party libraries
import glob
import numpy as np
# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# directs to custom libraries
import sys
sys.path.append('../src/')

# custom libraries
from ml import utils 


### User parameters
data_path_tmp = 'labeled/*.csv'
test_size = 0.25

# linear model
eta = 0.1
N_epochs = 20

# loads labeled data
data_paths = glob.glob(data_path_tmp)
for data_path in data_paths:
    # loads from csv, skipping header
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    # features
    X = data[:, :-1]
    # labeles
    y = data[:, -1]

    # splits into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    ### Linear model
    # trains
    w_start = np.random.randn(X_train.shape[1])
    W, losses = utils.SGD(X_train, y_train, w_start, eta, N_epochs)
    # tests
    y_pred = np.matmul(X_test, W[-1, :])
    rmse = np.sqrt( np.mean((y_test - y_pred)**2) )
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)

    print('RMSE = {0:.f}'.format(rmse))
    print('Precision = {0:.f}'.format(precision))
    print('Recall = {0:.f}'.format(recall))