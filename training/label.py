"""
label_data.py loads data from videos processed in main.py and sets up
a protocol for the user to classify objects identified in the videos
as true objects or noise.

Author: Andy Ylitalo
Date: September 29, 2021
"""

# standard libraries
import glob
import pickle as pkl
import csv
import os
import argparse

# 3rd party libraries
import cv2

# adds source folder
import sys
sys.path.append('../src/')

# imports custom libraries
from cvimproc import basic
from genl import fn
from classes.classes import Bubble



####################################################################################

### Global Constants ###
# fields to save
PROP_NAMES = ['area', 'major axis', 'minor axis', 'aspect ratio', 
                'orientation', 'solidity', 'inner stream', 'on border']

# header for output
HDR = PROP_NAMES + ['class']

# labeling message
MSG = 'Is this a true object? If so, press ''y''. Otherwise, press any other key.'
MAX_FIGSIZE = (1280, 960) # pixels; size of image of object


# creates argument parser for reading in user parameters
def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Classify objects and noise/artifacts from processed video data.')
    ap.add_argument('-u', '--unlabeled_path_tmp', default='unlabeled/*.pkl',
                    help='path template to unlabeled data')
    ap.add_argument('-o', '--overwrite', type=int, default=0,
                    help='If 1, overwrites existing labeled data')
    ap.add_argument('-d', '--save_dir', default='labeled',
                    help='Directory in which to save labeled data')
    ap.add_argument('-s', '--fig_scale', type=int, default=7,
                    help='Factor by which to scale figure size for OpenCV display')
    ap.add_argument('-e', '--ext', default='.csv',
                    help='Extension for save file')
    # loads arguments
    args = vars(ap.parse_args())

    # extracts and formats individual parameters
    unlabeled_path_tmp = args['unlabeled_path_tmp']
    overwrite = args['overwrite']
    save_dir = args['save_dir']
    fig_scale = args['fig_scale']
    ext = args['ext']

    return unlabeled_path_tmp, overwrite, save_dir, fig_scale, ext


### Functions ###

def scale_cv_figsize(im_shape, scale, max_figsize):
    """
    Scales image size for clearer display in OpenCV.
    
    Parameters
    ----------
    im_shape : 2-tuple of ints
        (number of rows, number of columns) in image to be scaled
    scale : float
        Positive factor to scale image size
    max_figsize : 2-tuple of ints
        (max number of rows, max number of columns) for image to be scaled
    
    Returns
    -------
    int : new x-dimension (number of columns)
    int : new y-dimension (number of rows)   
    """
    assert scale > 0
    return int( min(max_figsize[1], scale*im_shape[1]) ), \
           int( min(max_figsize[0], scale*im_shape[0]) )



### Analysis ###

def main():
    # loads user inputs
    unlabeled_path_tmp, overwrite, save_dir, fig_scale, ext = parse_args()

    # initializes output data structure (last column is object class)
    prop_vals = [[]]

    # globs input files
    unlabeled_paths = glob.glob(unlabeled_path_tmp)

    # loads data files
    for unlabeled_path in unlabeled_paths:
        
        # makes save name for labeled data
        save_name = os.path.split(unlabeled_path)[1][:-4] + ext
        save_path = os.path.join(save_dir, save_name)

        print('Loading data from {0:s}'.format(save_path))

        # skips this file if already labeled and not asked to overwrite data
        if os.path.isfile(save_path) and not overwrite:
            print('File exists and not asked to overwrite. Skipping.')
            continue

        # loads unlabeled data
        with open(unlabeled_path, 'rb') as f:
            data = pkl.load(f)

        # loads each object
        for ID, obj in data['objects'].items():
            frame_nums = obj.get_props('frame')
            
            # loads each observation of the object
            for n in frame_nums:
                # stores requested properties in list
                props = []
                for prop_name in PROP_NAMES:
                    props += [float(obj.get_prop(prop_name, n))]

                # extracts image of object
                im = obj.get_prop('image', n)
                # resizes and formats image (note: CV sizes are (x, y) not (row, col))
                figsize = scale_cv_figsize(im.shape, fig_scale, MAX_FIGSIZE)
                im_cv = cv2.resize(basic.cvify(im), figsize)

                # shows object and waits for user's key press
                cv2.imshow(MSG, im_cv)
                k = chr(cv2.waitKey(0))
                # true object if 'y' pressed
                is_obj = (k == 'y')

                # closes window
                cv2.destroyAllWindows()

                # stores features and classification
                prop_vals += [props + [float(is_obj)]]

        # saves as data for current file as csv with unique save name
        with open(save_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(HDR)
            write.writerows(prop_vals)


if __name__ == '__main__':
    main()