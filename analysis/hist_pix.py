"""
hist_pix.py is used to plot histograms of pixel values in raw 
and processed images. This may be helpful with gaining intuition
for appropriate thresholds to use for highlighting bubbles.

@author Andy Ylitalo
@date May 5, 2021
"""

# add source folder to path for imports
import sys
sys.path.append('../src/')

# imports 3rd-party libraries
import cv2

# imports custom libraries
import cvimproc.improc as improc
import cvimproc.basic as basic

### PARAMETERS ###
# path to video for testing
vid_path = '../input/sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_100_04_10.mp4'
th = 24
th_lo = 22
th_hi = 26
min_size_hyst = 2
min_size_th = 2
width_border = 2
sd = 2

### DERIVED PARAMETERS ###
selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd, sd))
mask_path = vid_path[:-4] + '_mask.pkl'

### COMPUTATIONS ###

# loads mask data
first_frame, _ = basic.load_frame(vid_path, 0)
mask_data = mask.get_polygonal_mask_data(first_frame, mask_path)

# Compute background


# load frames and process

# highlight bubbles and extract images at each step
#proc_ims = improc.highlight_bubble_hyst_thresh(
#                        frame, bkgd, th, th_lo, th_hi,
#                        min_size_hyst, min_size_th, width_border, 
#                        selem, mask_data, ret_all_steps=True)
#im_diff, thresh_bw_1, bubble_1, thresh_bw_2, bubble_2, bubble = proc_ims
