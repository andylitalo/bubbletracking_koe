"""true_obj_vs_th.py plots the number of "true" objects as a function of 
the threshold ('th' param). This plot will help me determine the appropriate threshold
for analysis.

Note: assumes that th_lo = th - 2 and th_hi = th + 2

Author: Andy Ylitalo
Date: February 24, 2022
"""

# standard libraries
import os 

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle as pkl

# custom libraries
import sys
sys.path.append('../src/')
import genl.fn as fn
import classes.classes as classes
import config as cfg

# PARAMETERS
vid_dir_list = ['ppg_co2/20211202_72bar/ppg_co2_40000_001-1_050_0319_83_04_21',
                'ppg_co2/20211202_72bar/ppg_co2_40000_001-1_050_0319_83_04_22']

### generally unchanged
subpath_tmp = 'th*/data/f_0_1_*.pkl'
# plot params
ax_fs = 16
# save subdirectory (in output directory)
save_subdir = 'analysis/true_obj_vs_th'
ext = 'svg' # file extension for plot

##################################################
# FUNCTIONS
def is_true(obj, true_props=['inner stream', 'oriented', 'consecutive', 'exited', 'centered']):
    """Returns True if true object and False if not."""
    for prop in true_props:
        # if lacks one of the key props for a true object, not a true object
        try:
            props = obj.get_props(prop)
            if False in props:
                return False
        except:
            print('DATA ARE MISSING TRUE PROPERTIES. Please rerun analysis.')
            continue
        
    return True 
#################################################


##############################################
# SCRIPT


for vid_dir in vid_dir_list:
    # data to plot
    th_list = []
    n_true_list = []

    # loads pkl data files
    filepath_tmp = os.path.join(cfg.output_dir, vid_dir, subpath_tmp)
    filepaths = glob.glob(filepath_tmp)
    for filepath in filepaths:
        # skips simplified data file for distribution
        if 'dist' in filepath:
            continue
        
        # loads data
        with open(filepath, 'rb') as f:
            data = pkl.load(f)

        # extracts threshold
        th = data['metadata']['highlight_kwargs']['th']

        # counts number of true objects
        n_true = 0

        for ID, obj in data['objects'].items():
            # counts true objects
            n_true += int(is_true(obj))

        # stores results for plotting
        th_list += [th]
        n_true_list += [n_true]

    # plots true objects vs. threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(th_list, n_true_list, 'ks')
    ax.set_xlabel('threshold', fontsize=ax_fs)
    ax.set_ylabel('# true objs', fontsize=ax_fs)

    # saves plot
    # extracts metadata of video
    params = fn.parse_vid_path(vid_dir)
    date, _, _ = fn.parse_vid_folder(vid_dir)
    if len(save_subdir) > 0:
        plt.savefig(os.path.join(cfg.output_dir, save_subdir, 
                        '{0:s}_{1:s}_{2:d}.{3:s}'.format(
                            params['prefix'], date, params['num'], ext)))

    # shows plot
    plt.show()

