"""
compare_outlier_metrics.py compares different metrics for outliers as a means
to identify appropriate thresholds for image segmentation to detect bubbles
with high fidelity.

The main idea is to assume that after subtracting the background, the remaining
pixel value distribution is composed of Gaussian noise centered around 0 and
pixels belonging to objects in the foreground. We use the parameters of the 
Gaussian distribution fitted to the pixels in object-free frames (in which
there should only be noise) to estimate thresholds for outliers.

The following methods for identifying outliers [1] are considered, wherein the 
statistics are computed from a Gaussian fit to the pixel values in a 
backgroun:
    1. > 3*sigma from the mean (sigma = standard deviation)
    2. > 5*sigma from the mean
    3. < Q1 - 1.5*IQR or > Q3 + 1.5*IQR (Q1 = 1st quartile, Q3 = 3rd quartile,
            IQR = interquartile range)
    4. modified Z-score > 3 (?)
        modified Z-score = 0.6745*(x_i - x_med) / (med(|x_i - x_med|)), 
        where x_i is the data point, x_med = med(x_i), and med() takes the 
        median [2].

Sources:
1. https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
2. Boris Iglewicz and David Hoaglin (1993), 
"Volume 16: How to Detect and Handle Outliers", The ASQC Basic References 
in Quality Control: Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

@author Andy Ylitalo
@date August 26, 2021
"""

### IMPORTS LIBRARIES
# adds path to source
import sys 
sys.path.append('../src/')

# standard libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 3rd party libraries 
import cv2 

# custom libraries 
import cvimproc.basic as basic 
import cvimproc.improc as improc 
import cvimproc.ui as ui
import cvimproc.mask as mask
import genl.fn as fn
import analysis.statlib as stat 


### PARAMETERS
# path to video for testing
vid_path = '../input/sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_04_10.mp4' 
# path to save folder
save_dir = '../output/analysis/compare_outlier_metrics/sd301_co2_40000_001_050_0280_88_04_10/'
# histogram plots
n_bins = 255
hist_max = 50
# analysis params
tol = 0.1 # requires mean to be close to 0
# frames to save (based on visual inspection of interesting frames in
# ANALYSIS\bubble-proc\sd301_co2\20210207_88bar\sd301_co2_15000_001_100_0335_79_100_04_10\hist_pix\below_mean
frames_to_save = [933, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4167, 
                    8939, 8940, 8941, 8942, 8943, 8944, 8945]

### DERIVED PARAMETERS ###
n_frames = basic.count_frames(vid_path)-1
mask_path = vid_path[:-4] + '_mask.pkl'

### ARGUMENTS TO PARSE ###
def parse_args():
    """Parses arguments for script."""
    ap = argparse.ArgumentParser(
        description='Plots histograms and saves image residuals' + \
                    ' where objects are detected.')
    ap.add_argument('-f', '--framerange', nargs=2, default=(0,100),
                    metavar=('frame_start, frame_end'),
                    help='starting and ending frames to save')
    ap.add_argument('-d', '--savediff', default=0, type=int,
                    help='Saves absolute difference from bkgd instead of image if True.')
    ap.add_argument('-c', '--colormap', default='',
                     help='Specifies colormap for saving false-color images.')
    ap.add_argument('-s', '--save', default=1, type=int,
                    help='Will only save images if True.')
    ap.add_argument('-x', '--ext', default='jpg',
                    help='Extension for saving images (no ".")')
    ap.add_argument('-n', '--num_frames_for_bkgd', default=1000, type=int,
                    help='Number of frames to average to compute background.')
    args = vars(ap.parse_args())

    frame_start, frame_end = int(args['framerange'][0]), int(args['framerange'][1])
    save_diff = args['savediff']
    colormap = args['colormap']
    save = args['save']
    ext = args['ext']
    num_frames_for_bkgd = args['num_frames_for_bkgd']

    return frame_start, frame_end, save_diff, colormap, \
            save, ext, num_frames_for_bkgd


########################## MAIN ##############################

def main():

    ### LOADS VIDEO AND PARAMS
    # reads args
    frame_start, frame_end, \
    save_diff, colormap, save, ext, num_frames_for_bkgd = parse_args()

    # loads mask data
    first_frame, _ = basic.load_frame(vid_path, 0)
    mask_data = ui.get_polygonal_mask_data(first_frame, mask_path)

    # Computes background
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
    bkgd = improc.compute_bkgd_med_thread(vid_path,
        vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
        num_frames=num_frames_for_bkgd)

    
    ### IDENTIFIES FRAMES WITHOUT OBJECTS (SYMM DISTR)
    symm_frame_list = []
    cap = cv2.VideoCapture(vid_path)
    for f in range(frame_start, frame_end):
        # loads frame
        frame = cv2.cvtColor(basic.read_frame(cap, f), cv2.COLOR_BGR2GRAY)
        # subtracts background, keeping sign 
        bkg_sub_sgn = frame.astype(int) - bkgd.astype(int)
        # checks for symmetric distribution (indicates no objs in foreground)
        if stat.is_symmetric(bkg_sub_sgn):
            symm_frame_list +=  [bkg_sub_sgn]

    ### ESTIMATES THRESHOLDS BASED ON ANALYSIS METHODS
    symm_frame_arr = np.asarray(symm_frame_list)
    mean = np.mean(symm_frame_arr)
    stdev = np.sqrt(np.mean(np.square(symm_frame_arr) - mean**2))
    # ensures that the distribution is indeed symmetric about 0
    assert np.abs(mean) < tol, 'mean of symmetric images too far from zero'

    # method 1: 3 sigma (loose req)
    th_3sig = int(3*stdev)
    # method 2: 5 sigma (strict req)
    th_5sig = int(5*stdev) 
    # method 3: IQR (more empirical, extreme version using 3*IQR)
    q1 = np.percentile(symm_frame_arr, 25)
    q3 = np.percentile(symm_frame_arr, 75)
    iqr = q3 - q1
    th_iqr = int(max(-q1 + 3*iqr, q3 + 3*iqr)) # takes max of 2 metrics
    ### MODIFIED Z-SCORE
    # sets a threshold for modified Z-score [2] as 3 or 5,
    # then solves for corresponding value (x_i in the formula)
    # method 4: modified z-score > 3
    median = np.median(symm_frame_arr)
    # median absolute deviation
    mad = np.median(np.abs(symm_frame_arr - median))
    th_mz3 = int(3 * (mad/0.6475 + median))
    # method 5: modified z-score > 5
    th_mz5 = int(5 * (mad/0.6475 + median))

    # prints results:
    print('mu = {0:f}, stdev = {1:f}, q1 = {2:f}, q3 = {3:f}'.format(mean, stdev, q1, q3))

    # collects thresholds and their names
    th_dict = {'3sig' : th_3sig, '5sig' : th_5sig, 'iqr' : th_iqr,
                'mz3' : th_mz3, 'mz5' : th_mz5}

    
    ### PLOTS HISTOGRAMS OF PIXEL VALUES IN SELECTED FRAMES ALONGSIDE THRESHOLDS

    # prepares output directory
    fn.makedirs_safe(save_dir)

    # loops through and saves desired frames
    for f in frames_to_save:
        frame = cv2.cvtColor(basic.read_frame(cap, f), cv2.COLOR_BGR2GRAY)
        # subtracts background, keeping sign 
        bkg_sub_sgn = frame.astype(int) - bkgd.astype(int)

        ### Plots histogram
        # creates figure
        plt.figure()
        _ = plt.hist(bkg_sub_sgn.flatten(), n_bins, histtype='step')
        # plots fitted Gaussian
        x_gauss = np.linspace(np.min(bkg_sub_sgn), np.max(bkg_sub_sgn), 1000)
        # computes height of gaussian at 0 by counting 0-valued pixels
        amplitude = len(np.where(bkg_sub_sgn.flatten() == 0)[0])
        y_gauss = amplitude * np.exp( -(x_gauss - mean)**2 / (2*stdev**2))
        plt.plot(x_gauss, y_gauss, 'k--', lw=2, label='gaussian fit')
        # plots thresholds
        for name, th in th_dict.items():
            plt.plot([-th, -th], [0, hist_max], '--', lw=2, label='{0:s} = {1:d}'.format(name, th))
        plt.title('Frame {0:d}, stdev = {1:.3f}'.format(f, stdev))
        # zooms in on the bottom of the distribution
        plt.ylim([0, hist_max])
        plt.legend()
        # saves figure
        if save:
            plt.savefig(os.path.join(save_dir, '{0:d}.{1:s}'.format(f, ext)))

    return


if __name__ == '__main__':
    main()