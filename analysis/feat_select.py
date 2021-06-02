"""
feat_select.py is a script that identifies the appropriate feature to select
for classification based on the comparison classification performance metrics,
such as the receiver-operator characteristic (ROC) curve and the area under the
curve (AUC).

This script will help me demonstrate why I choose one metric over another for 
bubble detection.

@author Andy Ylitalo
@date May 14, 2021
"""

# expand libraries to search to src folder
import sys
sys.path.append('../src/')

# Python libraries
import numpy as np

# local libraries
import cvimproc.basic as basic
import cvimproc.improc as improc
import statlib as stat

### USER-DEFINED PARAMETERS ###

# path to video
vid_path = '../input/sd301_co2/20210403_76bar/sd301_co2_40000_001_050_0280_88_04_10.mp4'   
# path to label data
data_dir = '../output/hist_pix/sd301_co2_40000_001_050_0280_88_04_10/'
true_bubbles_file = 'true_bubbles.csv'


### DERIVED PARAMETERS ###
# analysis parameters
num_frames_for_bkgd = basic.count_frames(vid_path)-1
end = basic.count_frames(vid_path)-1
mask_path = vid_path[:4] + '_mask.pkl'

# ??? APPLY MASK ???

# computes background (median)
bkgd = improc.compute_bkgd_med_thread(vid_path,
            vid_is_grayscale=True,
            num_frames=num_frames_for_bkgd)

# computes statistics of each frame
mean_list, mean_sq_list, stdev_list, min_val_list = stat.proc_stats(vid_path, bkgd, end)

# loads array of true bubbles
true_bubbles_path = data_dir + true_bubbles_file
true_bubbles = np.genfromtxt(true_bubbles_path, dtype=int, delimiter=',')
# produces binary vector of labels
y = stat.inds_to_labels(true_bubbles, end+1)

### COMPUTES ROC FOR DIFFERENT METRICS ###
# converts each metric so that higher values corespond to positive label
x_list = [-np.asarray(mean_list), 
            np.asarray(stdev_list), 
            -np.asarray(min_val_list)]
x_names = ['mean', 'stdev', 'min']

# computes and saves ROC curve for each metric
for x, name in zip(x_list, x_names):
    fpr, tpr, thresh = stat.thresh_roc(x, y)
    print(name)
    print(fpr)
    print(tpr)
    print(thresh)

    _ = stat.plot_roc(fpr, tpr, save_path=data_dir+name)
