"""
statlib.py is a library of functions useful for computing statistics of 
bubble classification and identification.

@author Andy Ylitalo
@date May 14, 2021
"""

# imports Python libraries
import numpy as np
import matplotlib.pyplot as plt

# imports 3rd party libraries
import cv2
from sklearn import metrics


def inds_to_labels(inds, n_samples):
    """
    Converts a list of indices of positive labels into a boolean vector,
    in which the indices of positive labels have 1s and the rest have 0s.

    Parameters
    ----------
    inds : list of ints
        indices of positive labels in the dataset
    n_samples : int
        number of samples in the dataset (must be larger than largest index)

    Returns
    -------
    labels : (n_samples x 1) array of bools
        1 for positive label, 0 otherwise
    """
    labels = np.zeros([n_samples])
    labels[inds] = 1    

    return labels.astype(bool) 

def plot_roc(fpr, tpr, save_path='', ext='.jpg'):
    """
    Plots receiver operating characteristic curve.

    Parameters
    ----------
    fpr : (N x 1) numpy array of floats
        False positive rate (FP/(FP+TN)) for different thresholds (can be unsorted)
    tpr : (N x 1) numpy array of floats
        True positive rate (TP/(TP+FN)) for different thresholds (must correspond to fpr)
    save_path : string, opt, default=''
        Path to save plot (includes directory). If '', not saved
    ext : string for 4 chars
        Extension for the desired image format of plot (includes '.')

    Returns
    -------
    ax : axis object
        Matplotlib axis of plot
    """
    # computes AUC
    inds_sort = np.argsort(fpr)
    fpr = fpr[inds_sort]
    tpr = tpr[inds_sort]
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC = {0:.5f}'.format(roc_auc))

    # plots ROC curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
        label='ROC curve (area = {0:.2f})'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, ls='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')

    # saves plot
    if len(save_path) > 0:
        plt.savefig(save_path + ext)

    return ax
    

def proc_stats(vid_path, bkgd, end, start=0):
    """
    Processes statistics of the images in the video.

    Parameters
    ----------
    vid_path : string
        path from current folder to video for analysis
    bkgd : (M x N) numpy array of uint8
        background of the video (usually median of several frames)
    end : int
        final frame to analyze
    start : int, opt (default = 0)
        first frame to analyze

    Returns
    -------
    mean_list : list of floats
        mean of the pixels in each frame analyzed (length = end - start)
    mean_sq_list : list of floats
        mean of the square of each pixel in each frame analyzed
        (length = end - start)
    stdev_list : list of floats
        standard deviation of the pixel values of each frame analyzed
        (length = end - start)
    min_val_list : list of floats
        minimum pixel value of each frame analyzed (length = end - start)
    """
    # initializes lists to store data
    mean_list = []
    stdev_list = []
    mean_sq_list = []
    min_val_list = []

    # load frames and process
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    f = start

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        signed_diff = frame.astype(int) - bkgd.astype(int)
        mean_list += [np.mean(signed_diff)]
        mean_sq_list += [np.mean(signed_diff**2)]
        stdev_list += [np.std(signed_diff)]
        min_val_list += [np.min(signed_diff)]

        if not ret or f >= end:
            break
        
        f += 1

    cap.release()

    return mean_list, mean_sq_list, stdev_list, min_val_list


def thresh_roc(x, y, n=100):
    """
    Computes the receiver-operator characteristic (ROC) curve given a 1D list
    of inputs and a 1D list of binary labels (0 or 1). Values in x are
    based on a threshold: y = 0 if x < thresh, and y = 1 if x >= thresh.

    Parameters
    ----------
    x : (N x 1) numpy array of floats
        input values (e.g., mean, minimum, standard deviation, etc.)
    y : (N x 1) numpy array of bools
        labels (0 or 1)
    n : int, opt (default = 100)
        number of thresholds to consider

    Returns
    -------
    thresh : (n x 1) numpy array of floats
        Thresholds on x used for classification, ordered from low to high
        and ranges from the minimum to the maximum value of x
    tpr : (n x 1) numpy array of floats
        True positive rate for each threshold in `thresh`
    fpr : (n x 1) numpy array of floats
        False positive rate for each threshold in `thresh`
    """
    # counts number of samples
    N = len(x)
    assert len(y) == N, 'samples x and labels y must have same length N.'
    # creates list of thresholds
    thresh = np.linspace(np.min(x), np.max(x), n)

    # computes true positive and false positive rates at each threshold
    tpr = np.zeros([n])
    fpr = np.zeros([n])
    # counts number of positives and negatives
    n_pos = np.sum(y)
    for i, th in enumerate(thresh):
        # makes predictions for given threshold
        y_pred = (x >= th)
        # counts true positives
        n_tp = np.dot(y.astype(int), y_pred.astype(int))
        # computes true positive rate
        tpr[i] = n_tp / n_pos
        # counts false positives
        n_fp = np.sum(y_pred) - n_tp
        # computes false positive rate
        fpr[i] = n_fp / (N - n_pos)

    return fpr, tpr, thresh
