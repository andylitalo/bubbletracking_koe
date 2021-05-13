"""
hist_pix.py is used to plot histograms of pixel values of the
signed difference from the background (median). It first computes
the mean and standard deviation of the signed difference from the
background, then uses that information to classify images based on 
mean, standard deviation, and minimum value of the signed difference.

Goal: for the classifications above to correspond to bubble detection.

@author Andy Ylitalo
@date May 5, 2021
"""

# add source folder to path for imports
import sys
sys.path.append('../src/')

# import Python libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# imports 3rd-party libraries
import cv2
import PIL.Image
import skimage.filters
from sklearn.cluster import KMeans

# imports custom libraries
import cvimproc.improc as improc
import cvimproc.basic as basic
import cvimproc.ui as ui
import cvimproc.mask as mask

### PARAMETERS ###
# path to video for testing
vid_path = '../input/sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_100_04_10.mp4'
# path to save folder
save_path = '../output/hist_pix/sd301_co2_15000_001_100_0335_79_100_04_10/'
# extension for images to save
ext = 'jpg'

num_frames_for_bkgd = 1000
end = basic.count_frames(vid_path)-1
# histogram plots
n_bins_mean = 100
n_bins_stdev = 100
n_bins_im = 255
hist_max = 20
# thresholds
th_mean = 0.25
th_stdev = 2.9
# number of standard deviations to require minimum
n_sigma = 6

### DERIVED PARAMETERS ###
mask_path = vid_path[:-4] + '_mask.pkl'

### COMPUTATIONS ###


# loads mask data
first_frame, _ = basic.load_frame(vid_path, 0)
mask_data = ui.get_polygonal_mask_data(first_frame, mask_path)

# Compute background
row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
bkgd = improc.compute_bkgd_med_thread(vid_path,
    vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
    num_frames=num_frames_for_bkgd)

"""
# Prints frames and histograms

for f in [9129, 9130]:
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im = PIL.Image.fromarray(frame)
    im.save(os.path.join(save_path, '{0:d}.{1:s}'.format(f, ext)))
    # plots histogram
    signed_diff = frame.astype(int) - bkgd.astype(int)
    plt.figure()
    _ = plt.hist(signed_diff.flatten(), n_bins_im, histtype='step')
    plt.ylim(0, hist_max)
    plt.title('{0:d}'.format(f))
    plt.savefig(os.path.join(save_path, 'hist_{0:d}.{1:s}'.format(f, ext)))
    plt.close()

sys.exit()
"""

# initializes lists to store data
mean_list = []
stdev_list = []
mean_sq_list = []
min_val_list = []

# load frames and process
cap = cv2.VideoCapture(vid_path)
f = 0

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

# plots histogram of mean and standard deviation
plt.figure()
_ = plt.hist(mean_list, n_bins_mean, histtype='step')
plt.title('Mean')
#plt.show()
plt.savefig(os.path.join(save_path, 'mean_hist.png'))

plt.figure()
_ = plt.hist(stdev_list, n_bins_stdev, histtype='step')
plt.title('Standard Deviation')
#plt.show()
plt.savefig(os.path.join(save_path, 'stdev_hist.png'))

####################### ROUND 2 OF ANALYSIS #############################
# now that we know the mean and standard deviation of each image, we can
# save images based on how their means and standard deviations compare to
# the overall values
print('Mean and standard deviation computed. Now save individual images and histograms.')

# average standard deviation
min_val_arr = np.asarray(min_val_list)
mu = np.mean(mean_list)
sigma = np.sqrt(np.mean(mean_sq_list) - mu**2)
sigma_approx = np.mean(stdev_list)
th_min = -n_sigma*sigma # removes mean since we want to compare to median (= 0)
# uses Otsu's method to identify threshold b/w min val for bubble and for no bubble
th_min_otsu = skimage.filters.threshold_otsu(min_val_arr)
# clusters minima into 2 -- use boundary as threshold?
inds_clusters = KMeans(n_clusters=2, random_state=1).fit_predict(min_val_arr.reshape(-1, 1))
i_low_cluster = inds_clusters[np.argmin(min_val_arr)]
th_min_kmeans_low = np.max(min_val_arr[inds_clusters==i_low_cluster])
i_high_cluster = inds_clusters[np.argmax(min_val_arr)]
th_min_kmeans_high = np.min(min_val_arr[inds_clusters==i_high_cluster])



# reports results
print('Mean mu = {0:f}, stdev sigma = {1:f}.'.format(mu, sigma))
print('Threshold for minimum is {0:.1f} for mu - {1:.1f}*sigma.' \
                        .format(th_min, n_sigma))
print('Approximate sigma is {0:f}.'.format(sigma_approx))
print('Otsu threshold for minimum is {0:.1f}'.format(th_min_otsu))
print('K-means low threshold for minimum is {0:.1f}'.format(th_min_kmeans_low))
print('K-means high threshold for minimum is {0:.1f}'.format(th_min_kmeans_high))


# clusters w/ kernal density estimation
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
# reshapes data for Kernel Density Estimation
X = min_val_arr.reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X)
# range of samples
samples = np.arange(-100, -10)
# expected density of samples
expected = kde.score_samples(samples.reshape(-1, 1))
# indices of minima and maxima of expected density
mi, ma = argrelextrema(expected, np.less)[0], argrelextrema(expected, np.greater)[0]
# minimum threshold is the highest minimum of the kernel density
th_min_kde = np.max(samples[mi])
print('Threshold from KDE is {0:.1f}.'.format(th_min_kde))


# load frames and process
cap = cv2.VideoCapture(vid_path)
f = 0

while(cap.isOpened()):
    # loads frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame, bkgd)
    
    signed_diff = frame.astype(int) - bkgd.astype(int)
    mean = np.mean(signed_diff)
    stdev = np.std(signed_diff)
    min_val = np.min(signed_diff)

    # saves image if a threshold is exceeded
    sub_dirs = []
    if mean > th_mean:
        sub_dirs += ['above_mean']
    if mean < -th_mean:
        sub_dirs += ['below_mean']
    if stdev > th_stdev:
        sub_dirs += ['above_stdev']
    if min_val < th_min:
        sub_dirs += ['below_min']

    # if both below min and above stdev, save in intersxn dir
    # this combo seems like a good test of bubbles
    if 'below_min' in sub_dirs and 'above_stdev' in sub_dirs:
        sub_dirs += ['stdev_min']

    for sub_dir in sub_dirs:
        print('saving frame {0:d}'.format(f))

        im = PIL.Image.fromarray(diff)
        im.save(os.path.join(save_path, sub_dir, '{0:d}.{1:s}'.format(f, ext)))
        # plotting as step is faster
        plt.figure()
        _ = plt.hist(signed_diff.flatten(), n_bins_im, histtype='step')
        plt.ylim(0, hist_max)
        plt.title('f{0:d}; mean = {1:.2f}; std = {2:.2f}; min = {3:d}'.format(f, \
                                    mean, stdev, int(min_val)))
        plt.savefig(os.path.join(save_path, sub_dir, \
                                    'hist_{0:d}.{1:s}'.format(f, ext)))
        plt.close()

    if not ret or f >= end:
        break
    
    f += 1

cap.release()


"""  
    # highlight bubbles and extract images at each step
    proc_ims = improc.highlight_bubble_hyst_thresh(
                            frame, bkgd, th, th_lo, th_hi,
                            min_size_hyst, min_size_th, width_border, 
                            selem, mask_data, ret_all_steps=True)
    
    im_diff, thresh_bw_1, bubble_1, thresh_bw_2, bubble_2, bubble = proc_ims
"""
