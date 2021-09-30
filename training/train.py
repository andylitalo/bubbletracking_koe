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

# directs to custom libraries
import sys
sys.path.append('../src/')

# custom libraries
from ml import utils 


### User parameters
data_path_tmp = 'labeled/*.csv'

# loads labeled data
data_paths = glob.glob(data_path_tmp)
for data_path in data_paths:
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]

    # splits into training and testing data

    ### Linear model

    # trains model

    # tests model