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
import glob

import sys
sys.path.append('../src/')
import config as cfg
import cvimproc.basic as basic
import genl.conversions as conv
import genl.readin as readin
import genl.main_helper as mh


def classify_test(obj, max_aspect_ratio=10, min_solidity=0.85,
            min_size=50, max_orientation=np.pi/8, circle_aspect_ratio=1.1,
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
                        'exited' : None, 'growing' : None}

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
    # TODO -- scale solidity requirement by size (currently lets through many non-solid large bubbles)
    classifications['solid'] = np.logical_not(np.logical_and(
                np.asarray(obj.get_props('solidity')) < min_solidity,
                worth_checking ))

    # too oblong?
    classifications['circular'] = np.logical_not(np.logical_and(
                np.asarray(obj.get_props('aspect ratio')) > max_aspect_ratio,
                np.asarray([row_max - row_min for row_min, _, row_max, _ in obj.get_props('bbox')]) \
                < obj.get_metadata('R_i')*conv.m_2_um*obj.get_metadata('pix_per_um') ))

    # oriented off axis (orientation of skimage.regionprops is CCW from up direction)
    classifications['oriented'] = np.logical_not(np.logical_and(
                np.logical_and(
                    np.abs(np.asarray(obj.get_props('orientation')) + np.pi/2) > max_orientation,
                    np.abs(np.asarray(obj.get_props('orientation')) - np.pi/2) > max_orientation),
                worth_checking ))

    # checks that the bubble appears in a minimum number of consecutive frames
    if obj.appears_in_consec_frames(n_consec=n_consec):
        classifications['consecutive'] = np.ones([len(obj.get_props('frame'))])
    else:
        classifications['consecutive'] = np.zeros([len(obj.get_props('frame'))])

    # flowing backwards
    classifications['inner stream'] = np.asarray(obj.get_props('flow speed [m/s]')) > obj.get_metadata('v_interf')

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
            classifications['exited'] = np.zeros([len(obj.get_props('frame'))])
        else:
            classifications['exited'] = np.ones([len(obj.get_props('frame'))])
    else:
        classifications['exited'] = np.zeros([len(obj.get_props('frame'))])

    # growing over time
    frames = np.asarray(obj.get_props('frame'))
    areas = np.asarray(obj.get_props('area'))
    if len(frames[worth_checking]) > 1:
        growing = np.polyfit(frames[worth_checking], areas[worth_checking], 1)[0] >= 0
    else:
        growing = True
    if growing:
        classifications['growing'] = np.ones([len(obj.get_props('frame'))])
    else:
        classifications['growing'] = np.zeros([len(obj.get_props('frame'))])

    return classifications


def load_objs(input_filepath, ext, save_dir):
    """
    Loads objects pointed to by input filepath.

    Parameters
    ----------
    input_filepath : string
        Path to input file (stored in `data` subfolder)
    ext : string
        extension of files saved before, without '.' (e.g., 'jpg')

    Returns
    -------
    objs : dictionary
        Dictionary of TrackedObj objects
    """
    # loads parameters
    p = readin.load_params(input_filepath)
    # determines appropriate directories based on parameters
    vid_path, data_path, vid_dir, \
    data_dir, figs_dir, _ = mh.get_paths(p)

    # determines save paths
    p['vid_subdir'] = save_dir
    _, _, _, _, figs_dir, _ = mh.get_paths(p)

    # if saving data, removes existing data
    files_to_remove = glob.glob(os.path.join(figs_dir, '*.{0:s}'.format(ext)))
    for filepath in files_to_remove:
        os.remove(filepath)

    # loads data
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
        objs = data['objects']

    return objs


def superpose_images(obj, classifications, skip_overlaps=False, prop='', color=(255,0,0)):
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
    M, N = obj.get_metadata('frame_dim')
    # initializes image black
    im = np.zeros([M, N, 3], dtype='uint8')

    # initializes previous bounding box
    bbox_prev = (0,0,0,0)

    # superposes image from each frame
    frame_list = obj.get_props('frame')
    for i, f in enumerate(frame_list):
        # loads bounding box and image within it
        bbox = obj.get_prop('bbox', f)

        # skips images that overlap if requested
        if skip_overlaps:
            if basic.is_overlapping(bbox_prev, bbox):
                continue
            else:
                bbox_prev = bbox

        # loads image and colors it
        im_obj_bw = obj.get_prop('image', f)
        m, n = im_obj_bw.shape
        im_obj = np.zeros([m, n, 3], dtype='uint8')

        # changes color to white if property required
        color_curr = color
        if len(prop) > 0:
            if classifications[prop][i]:
                color_curr = cfg.white 
        else:
            color_curr = cfg.white
        # colors image
        im_obj[im_obj_bw, :] = color_curr

        # superposes object image on overall image
        row_min, col_min, row_max, col_max = bbox
        im[row_min:row_max, col_min:col_max, :] = im_obj
    
    return im


###############################################################################

# OR, DO I WANT TO PROCESS THESE SPECIALLY?

def main():
    
    # input_subpaths = ['sd301_co2/20210331_45bar/sd301_co2_40000_001_050_0150_95_04_9/testclassify/data/input.txt',
    #                 'sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_04_10/testclassify/data/input.txt']

    input_subpaths = ['ppg_co2/20210720_70bar/ppg_co2_40000_001-1_050_0228_69_04_13/autothresh1/data/input.txt']
    ext = 'jpg'
    skip_overlaps = True
    save = True
    prop_list = ['solid', 'circular', 'oriented', ''] # default, no properties
    cmap = {'solid' : (255,0,0), 'circular' : (0,255,0), 'oriented' : (0,0,255),
                        'consecutive' : (255,255,0), 'inner stream' : (255,0,255),
                        'exited' : (0,255,255), 'growing' : (100, 250, 100), '' : (255,255,255)}
    save_dir = 'test/classification'

    input_filepaths = [os.path.join(cfg.output_dir, input_subpath) for input_subpath in input_subpaths]

    # loops through datasets
    for input_filepath in input_filepaths:
        print(input_filepath)

        objs = load_objs(input_filepath, ext, save_dir)

        # loops through objects
        for ID, obj in objs.items():
            # perform test classification
            classifications = classify_test(obj)
            # prints out classifications
            for prop, val in classifications.items():
                if any(np.logical_not(val)):
                    print('Object {0:d} is not {1:s}'.format(ID, prop))

            if save:
                for prop in prop_list:
                    # creates image of bubbles superposed on frame
                    im = superpose_images(obj, classifications, skip_overlaps=skip_overlaps, prop=prop, color=cmap[prop])
                    # saves image
                    basic.save_image(im, os.path.join(cfg.output_dir, save_dir, '{0:s}{1:d}.{2:s}'.format(prop, ID, ext)))    

    return 

if __name__ == '__main__':
    main()