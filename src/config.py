"""
config.py

Contains global variables and rarely changed parameters
for tracking bubbles.

Author: Andy Ylitalo
Date: April 29, 2021
"""

import numpy as np

import cvimproc.improc as improc

# directories
input_dir = '../input'
output_dir = '../output'
data_subdir = 'data'
figs_subdir = 'figs'

# image-processing methods
# method for highlighting objects
highlight_method = improc.highlight_obj_hyst_thresh
# method for assigning labels to objects
assign_method = improc.assign_objects
# object distance function (for labeling objects in object-tracking)
d_fn = improc.bubble_distance_v
                   
# colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# bubble filtering
def test_filter(objects):
    return True
    
filters = {0 : (None, {}),
            1 : (improc.filter_sph, {'min_size' : 12, 
                                    'min_solidity' : 0.9, 
                                    'max_angle' : np.pi/10}),
            2 : (test_filter, {})}


