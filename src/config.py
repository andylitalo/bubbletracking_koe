"""
config.py

Contains global variables and user-specific parameters
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
highlight_methods = {'highlight_bubble_hyst_thresh' :
                            improc.highlight_bubble_hyst_thresh}

# colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
