"""
label_data.py loads data from videos processed in main.py and sets up
a protocol for the user to classify objects identified in the videos
as true objects or noise.


"""

# standard libraries
import os
import glob
import pickle as pkl
import csv

# adds source folder
import sys
sys.path.append('../src/')

# imports custom libraries
from classes.classes import Bubble



####################################################################################

### User Parameters ###

# ***TODO change relative paths for absolute (?)

# input files (read txt from input dir?)
unlabeled_path_tmp = 'unlabeled/*.pkl'

# save directory and name
save_path = 'labeled/bubbles0.csv'

# fields to save
prop_names = ['area', 'major axis', 'minor axis', 'aspect ratio', 
                'orientation', 'solidity', 'inner stream', 'on border']

# header for output
hdr = prop_names + ['class']


### Analysis ###

# initializes output data structure (last column is object class)
prop_vals = [[]]

# globs input files
unlabeled_paths = glob.glob(unlabeled_path_tmp)

# loads data files
for unlabeled_path in unlabeled_paths:
    with open(unlabeled_path, 'rb') as f:
        data = pkl.load(f)

    # loads each object
    for ID, obj in data['objects'].items():
        frame_nums = obj.get_props('frame')
        
        for n in frame_nums:
            # stores features in array
            props = []
            for prop_name in prop_names:
                props += [float(obj.get_prop(prop_name, n))]

            # extracts image 
            im = obj.get_prop('image', n)

            # displays object for user's inspection

            # requests click from user to classify object
            # ? = true object; ? = noise/artifact
            is_obj = float(True)

        # stores features and classification
        prop_vals += [props + [is_obj]]

# saves as csv
with open(save_path, 'w') as f:
    write = csv.writer(f)
    write.writerow(hdr)
    write.writerows(prop_vals)
