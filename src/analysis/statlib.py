"""
statlib.py is a library of functions useful for computing statistics of 
object classification and identification.

@author Andy Ylitalo
@date May 14, 2021
"""

# imports Python libraries
import numpy as np
import matplotlib.pyplot as plt

# imports 3rd party libraries
import cv2
from sklearn import metrics
from sklearn.cluster import KMeans

# imports custom libraries
import cvimproc.mask as mask



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

def plot_roc(fpr, tpr, save_path='', ext='.jpg', show_fig=False,
            t_fs=18, ax_fs=16, tk_fs=14, l_fs=12):
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
    show_fig : bool, opt (default=False)
        If True, shows figure

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
        label='ROC curve (AUC = {0:.2f})'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, ls='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=ax_fs)
    ax.set_ylabel('True Positive Rate', fontsize=ax_fs)
    ax.set_title('Receiver Operating Characteristic', fontsize=t_fs)
    ax.legend(loc='lower right', fontsize=l_fs)
    ax.tick_params(axis='both', labelsize=tk_fs)
    if show_fig:
        plt.show()

    # saves plot
    if len(save_path) > 0:
        plt.savefig(save_path + ext)

    return ax
    

def proc_stats(vid_path, mask_data, bkgd, end, start=0, every=1,
                apply_mask=True, every_multiplier=100):
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
    every : int, optional
        Only loads every "every" frame (default = 1)
    apply_mask : bool, optional
        If True, will apply mask before computing statistics of a frame.
        Default True.
    every_multiplier : int, optional
        If `apply_mask` is `True`, will multiply `every` by this factor
        to speed up processing time (since applying the mask slows down
        computation). Default 100.

    Returns
    -------
    mean_list : list of floats
        mean of the pixels in each frame (bkgd-sub) analyzed (length = end - start)
    mean_sq_list : list of floats
        mean of the square of each pixel in each frame (bkgd_sub) analyzed
        (length = end - start)
    stdev_list : list of floats
        standard deviation of the pixel values of each frame (bkgd_sub) analyzed
        (length = end - start)
    min_val_list : list of floats
        minimum pixel value of each frame (bkgd-sub) analyzed (length = end - start)
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

    # loads rows to crop from mask
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)

    if apply_mask:
        # masks background
        bkgd = mask.mask_image(bkgd, mask_data['mask'][row_lo:row_hi])
        in_mask = np.ones(mask_data['mask'].astype(bool)[row_lo:row_hi, :].shape, dtype=int)
        every *= every_multiplier

    while(cap.isOpened()):
        # skips every "every" number of frames
        if (f - start) % every != 0:
            f += 1
            continue

        # reads current frame
        ret, frame = cap.read()

        # stops at video or user-specified end frame
        if not ret or f >= end:
            break
        
        # converts frame to grayscale and masks
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame_gray #mask.mask_image(frame_gray, mask_data['mask'])
        # crops frame
        frame_cropped = frame[row_lo:row_hi, :]

        # loads data for statistics
        signed_diff = frame_cropped.astype(int) - bkgd.astype(int)

        if not apply_mask:
            # ensures only evaluates points in the mask
            mean_list += [np.mean(signed_diff)]
            mean_sq_list += [np.mean(signed_diff**2)]
            stdev_list += [np.std(signed_diff)]
            min_val_list += [np.min(signed_diff)]
        else:
            # in_mask = np.ones(mask_data['mask'].astype(bool)[row_lo:row_hi, :].shape, dtype=int)
            mean_list += [np.mean(signed_diff[in_mask])]
            mean_sq_list += [np.mean(signed_diff[in_mask]**2)]
            stdev_list += [np.std(signed_diff[in_mask])]
            min_val_list += [np.min(signed_diff[in_mask])]

        # increments frame number
        f += 1

    cap.release()

    return mean_list, mean_sq_list, stdev_list, min_val_list


def suggest_thresholds(vid_path, mask_data, bkgd, start, end, every,
                        th_reduction=0.9, n_stdev=5):
    """
    Suggests high and low thresholds for hysteresis threshold and 
    threshold for uniform threshold based on k-means and other stats.

    Created and tested in `analysis/hist_pix.py`.

    Parameters
    ----------

    Returns
    -------
    th, th_lo, th_hi : uint8
        Uniform threshold, low threshold for hysteresis thresholding,
        and high threshold for hysteresis thresholding
    """
    # computes statistics of the image
    mean_list, mean_sq_list, \
    stdev_list, min_val_list = proc_stats(vid_path, mask_data, bkgd, end, 
                                            start=start, every=every)
    # converts to array for numpy operations
    min_val_arr = np.asarray(min_val_list)
    # zeros positive values and switches sign to focus on bubbles
    min_val_arr[min_val_arr > 0] = 0
    min_val_arr *= -1
    # divides minima into 2 clusters with k-means algorithm
    inds_clusters = KMeans(n_clusters=2, 
                        random_state=1).fit_predict(min_val_arr.reshape(-1, 1))
    # identifies the boundaries of the clusters to use as suggested thresholds
    i_low_cluster = inds_clusters[np.argmin(min_val_arr)]
    max_low_cluster = np.max(min_val_arr[inds_clusters==i_low_cluster])
    i_high_cluster = inds_clusters[np.argmax(min_val_arr)]
    min_high_cluster = np.min(min_val_arr[inds_clusters==i_high_cluster])

    # sets uniform threshold to upper bound of lower k-means cluster
    th = int(np.abs(max_low_cluster))
    # sets high hysteresis threshold to lower bound of higher k-means cluster
    # *same as Otsu's method
    th_hi = int(np.abs(min_high_cluster))

    # finds frames of pure background and computes their mean and standard deviation
    i_bkgd = np.where(min_val_arr < th_reduction*th)

    mean_bkgd = np.mean(np.asarray(mean_list)[i_bkgd])
    stdev_bkgd = np.mean(np.asarray(stdev_list)[i_bkgd])
    # computes low threshold as some standard deviations above noise
    th_lo = int(mean_bkgd + n_stdev*stdev_bkgd)

    # # sets low hysteresis threshold to mean of minimum + 1 standard deviation
    # th_lo = int(np.abs(np.mean(min_val_arr)) + np.std(min_val_arr))

    # checks that thresholds are in proper order--warns and sorts if not
    if th_lo > th or th > th_hi:
        print('suggest_thresholds() yielded unordered thresholds:')
        print('th_lo = {0:d}, th = {1:d}, th_hi = {2:d}'.format(th_lo, th, th_hi))
        print('Reordering and returning, but values are *not recommended*.')
        thresholds = [th_lo, th, th_hi]
        thresholds.sort()
        th_lo, th, th_hi = thresholds

    return th, th_lo, th_hi


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
