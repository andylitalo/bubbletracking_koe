"""
utils.py contains useful functions for machine learning. Most of these functions
come from my homework submissions to CS 155, taught in Winter 2019 at Caltech by
Prof. Yisong Yue.

Author: Andy Ylitalo
Date: September 29, 2021
"""

# 3rd party libraries
import numpy as np


def loss(X, Y, w):
    """
    Calculate the squared loss function.
    
    Parameters
    ----------
    X : A (N, D) shaped numpy array 
        Data points
    Y : A (N, ) shaped numpy array
        (float) labels of the data points
    w : A (D, ) shaped numpy array
        weight vector
    
    Returns
    -------
    (float) The loss evaluated with respect to X, Y, and w.
    """
    # compute predicted output using linear model (no bias)
    Y_pred = X.dot(w)
    
    # compute squared loss
    sq_loss = np.sum((Y_pred - Y)**2)
    
    return sq_loss


def gradient(x, y, w):
    """
    Calculate the gradient of the loss function with respect to the weight vector w,
    evaluated at a single point (x, y) and weight vector w.
    
    Parameters
    ----------
    x : A (D, ) shaped numpy array
        single data point.
    y : float
        label for the data point.
    w : A (D, ) shaped numpy array
        weight vector.
        
    Returns
    -------
    (float) The gradient of the loss with respect to w. 
    """
    # computes gradient based on result from Problem 4B
    return -2*x*(y - np.dot(w, x))


def SGD(X, Y, w_start, eta, N_epochs):
    """
    Performs stochastic gradient descent (SGD) using dataset (X, Y), 
    initial weight vector w_start, learning rate eta, and N_epochs epochs.
    
    Parameters
    ----------
    X : A (N, D) shaped numpy array of floats
        data points.
    Y : A (N, ) shaped numpy array of floats
        labels of the data points.
    w_start :  A (D, ) shaped numpy array of floats 
        weight vector initialization.
    eta : float
        The step size.
    N_epochs : int
        The number of epochs (iterations) to run SGD.
        
    Returns
    -------
    W : A (N_epochs, D) shaped array of floats
        weight vectors from all iterations.
    losses : A (N_epochs, ) shaped array of floats
        losses from all iterations.
    """
    # get number of data points and dimension of data
    N_pts, D = np.shape(X)
    
    # initialize outputs
    W = np.zeros([N_epochs, D])
    losses = np.zeros(N_epochs)
    
    # update outputs based on initial weights
    W[0,:] = np.copy(w_start)
    losses[0] = loss(X, Y, w_start)
    
    # store initial weights
    w = np.copy(w_start)
    
    # loop through N_epochs, where first epoch is initial weights
    for i in range(1, N_epochs):
        
        # shuffle indices
        inds_shuffled = np.random.permutation(np.arange(N_pts))

        # loop through data points in random order
        for j in inds_shuffled:
            
            # get data point and output
            x = X[j,:]
            y = Y[j]
            
            # compute gradient
            grad = gradient(x, y, w)
            
            # update weights
            w -= eta*grad         
        
        # store new weights after each epoch
        W[i,:] = w
        # measure loss after each epoch and store
        losses[i] = loss(X, Y, w)
        
    return W, losses