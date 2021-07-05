"""
test_classification.py tests the success of my classification 
method Bubble.classify() by using a modified version added to 
the test version of the object, BubbleTest.classify_test(), which
indicates which level of classification the object passed (looks like
a bubble) and failed (does not look like a bubble).

***Run this on many datasets until you hone in the classification parameters***
OR, write an ML algo to perform this (requires efficient labeling pipeline)

Author: Andy Ylitalo
Date: July 5, 2021
"""

import os
import pickle as pkl
import numpy as np

import sys
sys.path.append('../src/')
import config as cfg
import genl.conversions as conv


def classify_test(obj, max_aspect_ratio=10, min_solidity=0.9,
            min_size=10, max_orientation=np.pi/10, circle_aspect_ratio=1.1,
            n_consec=3):
    """
    Classifies Bubble objects into different categories. Still testing to see
    what classifications are most important.

    Turn this into ML with the following labels (sequences of):
    - Aspect Ratio
    - Orientation
    - Area
    - Rate of area growth
    - Speed along flow direction
    - Solidity
    - *averages thereof
    - predicted max speed
    - width of inner stream
    - height

    ***Ask Chris about this! TODO
    """
    # creates dictionary of classifications
    classifications = {'solid' : None, 'circular' : None, 'oriented' : None,
                        'consecutive' : None, 'inner stream' : None}

    # computes average speed [m/s] if not done so already
    if obj.get_props('average flow speed [m/s]') == None:
        obj.process_props()

    # first checks if object is an artifact (marked by inner_stream = -1)
    # not convex enough?
    if any( np.logical_and(
                np.asarray(obj.get_props('solidity')) < min_solidity,
                np.asarray(obj.get_props('area')) > min_size )
            ):
        classifications['solid'] = False 
    else:
        classifications['solid']= True 

    # too oblong?
    if any( np.logical_and(
                np.asarray(obj.get_props('aspect ratio')) > max_aspect_ratio,
                np.asarray([row_max - row_min for row_min, _, row_max, _ in obj.get_props('bbox')]) \
                < obj.get_metadata('R_i')*conv.m_2_um*obj.get_metadata('pix_per_um') )
        ):
        classifications['circular'] = False 
    else:
        classifications['circular'] = True 

    # oriented off axis (orientation of skimage.regionprops is CCW from up direction)
    if any( np.logical_and(
                np.abs(np.asarray(obj.get_props('orientation')) + np.pi/2) < max_orientation,
                np.asarray(obj.get_props('area')) > min_size )
            ):
        classifications['oriented'] = False
    else:
        classifications['oriented'] = True 

    # checks that the bubble appears in a minimum number of consecutive frames
    if not obj.appears_in_consec_frames(n_consec=n_consec):
        classifications['consecutive'] = False 
    else:
        classifications['consecutive'] = True

    # flowing backwards or flowing too fast
    if any( np.logical_or(
                np.asarray(obj.get_props('flow speed [m/s]')) < 0,
                np.asarray(obj.get_props('speed [m/s]') > 3*obj.get_metadata('v_max')) )
        ):
        classifications['inner stream'] = None
    elif any( np.asarray(obj.get_props('flow speed [m/s]')) > obj.get_metadata('v_interf')):
        classifications['inner stream'] = True
    else:
        classifications['inner stream'] = False

    return classifications




# OR, DO I WANT TO PROCESS THESE SPECIALLY?

def main():
    vid_subdirs = [#'sd301_co2/20210331_45bar/sd301_co2_40000_001_050_0150_95_04_9/',
                    'sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_04_10/testclassify/data/f_0_1_11089.pkl']

    # creates list of dataset filepaths
    data_filepaths = [os.path.join(cfg.output_dir, vid_subdir) for vid_subdir in vid_subdirs]

    # loops through datasets
    for data_filepath in data_filepaths:
        print(data_filepath)

        # loads data
        with open(data_filepath, 'rb') as f:
            data = pkl.load(f)
            objects = data['objects']

        # loops through objects
        for ID, obj in objects.items():
            # perform test classification
            classifications = classify_test(obj)



    # creates image of bubbles superposed on frame

    # how do I record results of classification? preferably on image?

    # save image

    return 

if __name__ == '__main__':
    main()