"""bubbles.py

Analyzes data about bubbles related to growth and size..

Bubble data are saved in pickle files indexed by bubble IDs.

Date: March 7, 2021
Author: Andy Ylitalo
"""
import sys
sys.path.append('../src/')

import pickle as pkl

data_dir = '../output/data/20210207_88bar/sd301_co2_15000_001_100_0335_79_100_04_10/hi_2/'
data_file = 'f_0_1_11089.pkl'

with open(data_dir + data_file, 'rb') as f:
    data = pkl.load(f)

data_to_save = {}

for ID in data['bubbles']:

    bubble = data['bubbles'][ID]

    centroids = bubble.get_props('centroid')
    bboxes = bubble.get_props('bbox')
    on_border = bubble.get_props('on border')
    frames = bubble.get_props('frame')
    fps = bubble.get_metadata('fps')
    pix_per_um = 1/2.29 # known value

    length = []
    width = []
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
            t += [(frames[i] - frames[0])/fps]
            rc, cc = centroids[i]
            leading += [max_col/pix_per_um]
            trailing += [min_col/pix_per_um]
            center += [cc/pix_per_um]

    data_to_save[ID] = {'t' : t, 'length' : length, 'width' : width, 
                    'leading' : leading, 'trailing' : trailing,
                    'center' : center}

with open(data_dir + 'bubble_data.pkl', 'wb') as f:
    pkl.dump(data_to_save, f)
