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
import cvimproc.basic as basic
import genl.conversions as conv
import genl.readin as readin
import genl.main_helper as mh


def classify_test(obj, max_aspect_ratio=10, min_solidity=0.85,
            min_size=20, max_orientation=np.pi/8, circle_aspect_ratio=1.1,
            n_consec=3):
    """
    Classifies Bubble objects into different categories. Still testing to see
    what classifications are most important.

    Parameters are currently guesses. ML would be a better method for selecting them.

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
    ***CHALLENGE: how do I classify large, elongated bubbles separately? TODO
    """
    # creates dictionary of classifications
    classifications = {'solid' : None, 'circular' : None, 'oriented' : None,
                        'consecutive' : None, 'inner stream' : None,
                        'exited' : None}

    # computes average speed [m/s] if not done so already
    if obj.get_props('average flow speed [m/s]') == None:
        obj.process_props()

    # determines which frames are worth looking at (large enough object, not on border)
    worth_checking = np.logical_and(
                            np.asarray(obj.get_props('area')) > min_size,
                            np.logical_not(np.asarray(obj.get_props('on border')))
                            )

    # first checks if object is an artifact (marked by inner_stream = -1)
    # not convex enough?
    if any( np.logical_and(
                np.asarray(obj.get_props('solidity')) < min_solidity,
                worth_checking )
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
                np.logical_and(
                    np.abs(np.asarray(obj.get_props('orientation')) + np.pi/2) > max_orientation,
                    np.abs(np.asarray(obj.get_props('orientation')) - np.pi/2) > max_orientation),
                worth_checking )
            ):
        classifications['oriented'] = False
    else:
        classifications['oriented'] = True 

    # checks that the bubble appears in a minimum number of consecutive frames
    if not obj.appears_in_consec_frames(n_consec=n_consec):
        classifications['consecutive'] = False 
    else:
        classifications['consecutive'] = True

    # flowing backwards
    if any(np.asarray(obj.get_props('flow speed [m/s]')) < 0):
        classifications['inner stream'] = None
    elif any( np.asarray(obj.get_props('flow speed [m/s]')) > obj.get_metadata('v_interf')):
        classifications['inner stream'] = True
    else:
        classifications['inner stream'] = False

    # exits properly from downstream side of frame
    # farthest right column TODO -- determine coordinate of bbox from flow dir
    col_max = obj.get_props('bbox')[-1][-1] 
    # average flow speed [m/s]
    average_flow_speed_m_s = obj.get_props('average flow speed [m/s]') 
    # number of pixels traveled per frame along flow direction
    if average_flow_speed_m_s is not None:
        pix_per_frame_along_flow = average_flow_speed_m_s * conv.m_2_um * \
                        obj.get_metadata('pix_per_um') / obj.get_metadata('fps')
        # checks if the next frame of the object would reach the border or if the 
        # last view of object is already on the border
        if (col_max + pix_per_frame_along_flow < obj.get_metadata('frame_dim')[1]) and \
                not obj.get_props('on border')[-1]:
            classifications['exited'] = False 
        else:
            classifications['exited'] = True
    else:
        classifications['exited'] = False

    return classifications


def is_overlapping(bbox1, bbox2):
    """
    Checks if the bounding boxes given overlap.

    Copied from https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    rmin1, cmin1, rmax1, cmax1 = bbox1
    rmin2, cmin2, rmax2, cmax2 = bbox2

    # To check if either rectangle is actually a line
    # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
    if (cmin1 == cmax1 or rmin1 == rmax1 or cmin2 == cmax2 or rmin2 == rmax2):
        # the line cannot have positive overlap
        return False
       
     
    # If one rectangle is on left side of other
    if(cmin1 >= cmax2 or cmin2 >= cmax1):
        return False
 
    # If one rectangle is above other
    if(rmin1 >= rmax2 or rmin2 >= rmax1):
        return False
 
    return True


def superpose_images(obj, skip_overlaps=False):
    """
    Superposes images of an object onto one frame.

    Parameters
    ----------
    obj : TrackedObject
        Object that has been tracked. Must have 'image', 'local centroid',
        'frame_dim', 'bbox', and 'centroid' parameters.

    Returns
    -------
    im : (M x N) numpy array of uint8
        Image of object over time; each snapshot is superposed 
        (likely black-and-white)
    """
    # gets dimensions of frame
    frame_dim = obj.get_metadata('frame_dim')
    # initializes image black
    im = np.zeros(frame_dim, dtype='uint8')

    # initializes previous bounding box
    bbox_prev = (0,0,0,0)

    # superposes image from each frame
    frame_list = obj.get_props('frame')
    for f in frame_list:
        # loads bounding box and image within it
        bbox = obj.get_prop('bbox', f)

        # skips images that overlap if requested
        if skip_overlaps:
            if is_overlapping(bbox_prev, bbox):
                continue
            else:
                bbox_prev = bbox

        # loads image
        im_obj = obj.get_prop('image', f)

        # superposes object image on overall image
        row_min, col_min, row_max, col_max = bbox
        im[row_min:row_max, col_min:col_max] = basic.cvify(im_obj)
    
    return im






# OR, DO I WANT TO PROCESS THESE SPECIALLY?

def main():
    
    input_subpaths = ['sd301_co2/20210331_45bar/sd301_co2_40000_001_050_0150_95_04_9/testclassify/data/input.txt',
                    'sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_04_10/testclassify/data/input.txt']
    ext = 'jpg'
    replace = True
    skip_overlaps = True

    input_filepaths = [os.path.join(cfg.output_dir, input_subpath) for input_subpath in input_subpaths]

    # loops through datasets
    for input_filepath in input_filepaths:
        print(input_filepath)

        # loads parameters
        p = readin.load_params(input_filepath)
        # determines appropriate directories based on parameters
        vid_path, data_path, vid_dir, \
        data_dir, figs_dir, _ = mh.get_paths(p, replace)

        # determines save paths
        p['vid_subdir'] = 'test/classification/'
        _, _, _, _, figs_dir, _ = mh.get_paths(p, replace)

        # loads data
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
            objects = data['objects']

        # loops through objects
        for ID, obj in objects.items():
            # perform test classification
            classifications = classify_test(obj)
            # prints out classifications
            for prop, val in classifications.items():
                if not val:
                    print('Object {0:d} is not {1:s}'.format(ID, prop))

            # creates image of bubbles superposed on frame
            im = superpose_images(obj, skip_overlaps=skip_overlaps)
            # saves image
            basic.save_image(im, os.path.join(figs_dir, '{0:d}.{1:s}'.format(ID, ext)))    

    return 

if __name__ == '__main__':
    main()