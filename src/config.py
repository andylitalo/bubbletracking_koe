"""
config.py

Contains global objects and user-specific parameters
for tracking bubbles.

@author: Andy Ylitalo
@date: April 29, 2021
"""

import cvimproc.improc as improc

# directories
input_dir = '../input'
output_dir = '../output'
data_subdir = 'data'
figs_subdir = 'figs'

# highlight methods
highlight_methods = {'highlight_obj_hyst_thresh' :
                            improc.highlight_obj_hyst_thresh,
                    'highlight_obj_hyst' :
                            improc.highlight_obj_hyst}

# colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
