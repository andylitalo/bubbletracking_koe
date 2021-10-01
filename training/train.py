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

# standard libraries
import pickle as pkl

# 3rd party libraries
import glob
import numpy as np
# ML
from sklearn.model_selection import train_test_split # better than PyTorch's random_split 
# as of 11/26/20 -- see: https://github.com/PyTorchLightning/lightning-bolts/issues/312
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# directs to custom libraries
import sys
sys.path.append('../src/')

# custom libraries
from ml import utils 

##############################################################################


### User parameters
data_path_tmp = 'labeled/*.csv'
test_size = 0.1
# seeds random selections in ML pipeline
random_state = 1
# random forest depth
max_depth = 4
# path to save models
save_path = 'trained_models.pkl'


### Functions ###

# def linear_model(x):
#     return x @ w.t() + b

# def mse(t1, t2):
#     """Computes the mean-squared error"""
#     diff = t1 - t2 
#     return torch.sum(diff * diff) / diff.numel()


### Computation ###

# loads labeled data
data_paths = glob.glob(data_path_tmp)
for data_path in data_paths:
    # loads from csv, skipping header
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

    ############################ PREPROCESS DATA #####################################
    # features
    X = data[:, :-1]
    # labeles
    y = data[:, -1]

    # splits into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state)

    ############################### TRAIN AND TEST ######################################
    
    
    ### Logistic Regression ###
    # trains model after standardizing input
    clf_log = make_pipeline(StandardScaler(),
                    SGDClassifier(loss='log', penalty='l1', 
                    early_stopping=True, random_state=random_state)
                    )
    clf_log.fit(X_train, y_train)

    # scores model on test data
    accuracy = clf_log.score(X_test, y_test)
    print('Logistic Regression: accuracy = {0:.3f}'.format(accuracy))

    print('Coeffs for logistic regression:')
    print(clf_log.coef_)
    print()

    ### Linear SVM ###
    # trains model after standardizing input
    clf_svm = make_pipeline(StandardScaler(),
                        SVC(kernel='linear', random_state=random_state)
                        )
    clf_svm.fit(X_train, y_train)

    # scores model on test data
    accuracy = clf_svm.score(X_test, y_test)
    print('SVM: accuracy = {0:.3f}'.format(accuracy))


    ### Random Forest ###
    # trains model after standardizing input
    clf_rf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(max_depth=max_depth, 
                            random_state=random_state)
                            )
    clf_rf.fit(X_train, y_train)

    # scores model on test data
    accuracy = clf_rf.score(X_test, y_test)
    print('Random Forest: accuracy = {0:.3f}'.format(accuracy))

   
    ######################## STORE MODELS ##############################
    
    # pickle 
    pkl.dump(clf_rf, open(save_path, 'wb'))

    # unpickle
    clf_rf_reloaded = pkl.load(open(save_path, 'rb'))

    # test accuracy
    accuracy = clf_rf_reloaded.score(X_test, y_test)
    print('Random forest (reloaded): accuracy = {0:.3f}'.format(accuracy))



    ### PYTORCH IMPLEMENTATION -- waiting to install pytorch ###
    # based on: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
    # trains
    # X_train = torch.from_numpy(X_train)
    # y_train = torch.from_numpy(y_train)
    # # initializes weights and biases
    # w = torch.randn(X_train.shape[1], requires_grad=True)
    # b = torch.randn(1, requires_grad=True)

    # # trains for epochs
    # for i in range(N_epochs):
    #     y_pred = model(X_train)
    #     loss = mse(y_pred, y_train)
    #     loss.backward()
    #     with torch.no_grad():
    #         w -= w.grad() * eta 
    #         b -= b.grad() * eta 
    #         w.grad.zero_()
    #         b.grad.zero_()



    # # calculates final loss
    # y_pred = model(X_train)
    # loss = mse(y_pred, y_train)
    # print(loss)

    
    # w_start = np.random.randn(X_train.shape[1])
    # W, losses = utils.SGD(X_train, y_train, w_start, eta, N_epochs)
    # # tests
    # y_pred = np.matmul(X_test, W[-1, :])
    # rmse = np.sqrt( np.mean((y_test - y_pred)**2) )
    # precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)

    # print('RMSE = {0:.f}'.format(rmse))
    # print('Precision = {0:.f}'.format(precision))
    # print('Recall = {0:.f}'.format(recall))