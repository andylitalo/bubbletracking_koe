# training

This folder contains methods used to train a machine-learning model
to distinguish bubbles from noise and artifacts.

## Current Status

November 11, 2021: 
Logistic Regression: accuracy = 0.890
SVM: accuracy = 0.795
Random Forest: accuracy = 0.932

Since classification error is pretty high with ML models, I will try
classifying manually (setting thresholds) because that is easier to
understand and implement in the C++ code.

I should try PCA first just to see the spread in the data