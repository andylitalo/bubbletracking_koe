"""thresh_bubbles.py

Analyzes data about bubbles related to growth and size for
different thresholds applied to the image processing method.

Data are stored in a dictionary indexed by the name of the 
threshold set (often the value of the plain threshold).
The entry for a threshold contains a dictionary of the
settings of the threshold and a list of dictionaries of the
properties of each bubble.

Because I cannot install Atom or Jupyter notebook easily
on this Linux subsystem of Windows, I save the data to my 
Windows partition for plotting and analysis there.

Date: April 28, 2021
Author: Andy Ylitalo
"""

import pickle as pkl
import os

import sys
sys.path.append('../src')

from classes.classes import Bubble

# parameters
# directory to folders for different input parameters
output_dir = '../output/'
# path from output directory to video directory
vid_subdir = 'sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_100_04_10/'
# sub folder containing data
data_folder = 'data'
# name of the data file to process
data_file = 'f_0_1_11089.pkl'
# folder to save to
save_dir = '/mnt/c/Users/andyl/OneDrive - California Institute ' + \
            'of Technology/Documents/Research/Kornfield/ANALYSIS/bubble-proc/' + \
            vid_subdir + 'test_thresh/'

# builds data directory
data_dir = output_dir + vid_subdir

# extracts data for bubbles highlighted by each threshold set
for entry in os.scandir(data_dir):

    # skips entries that are not directories
    if not entry.is_dir():
        continue
    # creates and opens path to data for current threshold set
    thresh_path = os.path.join(entry.path, data_folder, data_file)
    with open(thresh_path, 'rb') as f:
        data = pkl.load(f)

    # creates dictionary to store data for bubbles
    bubbles = {}

    # collects relevant data for each bubble
    for ID in data['bubbles']:

        # extracts bubble object from data dictionary
        bubble = data['bubbles'][ID]
        # extracts relevant properties from bubble object
        area = bubble.get_props('area')
        centroids = bubble.get_props('centroid')
        bboxes = bubble.get_props('bbox')
        on_border = bubble.get_props('on border')
        frames = bubble.get_props('frame')
        fps = bubble.get_metadata('fps')
        pix_per_um = bubble.get_metadata('pix_per_um') # known value

        # initializes lists to store additional data
        length = []
        width = []
        radius = []
        t = []
        leading = []
        trailing = []
        center = []

        for i in range(len(bboxes)):
            if not on_border[i]:
                bbox = bboxes[i]
                min_row, min_col, max_row, max_col = bbox
                length += [(max_col - min_col)/pix_per_um]
                width += [(max_row - min_row)/pix_per_um]
                # estimates radius by assuming axisymmetry along flow axis
                radius += [(length[-1]*width[-1]**2)**(1.0/3)/2]
                t += [(frames[i] - frames[0])/fps]
                rc, cc = centroids[i]
                leading += [max_col/pix_per_um]
                trailing += [min_col/pix_per_um]
                center += [cc/pix_per_um]

        bubbles[ID] = {'t' : t, 'length' : length, 'width' : width, 
                        'leading' : leading, 'trailing' : trailing,
                        'center' : center, 'centroids' : centroids, 
                        'area' : area}

    # extracts parameters for current thresholds
    p = data['metadata']['input params']
    # only stores relevant threshold parameters to avoid pickling cvvidproc lib
    params = {'th' : p['th'], 'th_lo' : p['th_lo'], 'th_hi' : p['th_hi']}
    # saves data to dictionary
    data_to_save = {'name': entry.name, 'params' : params, 'bubbles' : bubbles}

    with open(save_dir + entry.name + '_' + data_file, 'wb') as f:
        pkl.dump(data_to_save, f)

