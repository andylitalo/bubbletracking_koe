"""
lib.py contains functions for tests.

Author: Andy Ylitalo
Date: January 29, 2021
"""

# imports standard libraries
import os
import time
import numpy as np

# imports 3rd-party libraries
import cv2

# adds filepath to custom libraries
import sys
sys.path.append('../src/')
# imports custom libraries
import cvimproc.improc as improc
import cvimproc.basic as basic
import genl.readin as readin
import main
import config as cfg




def test_hysteresis_threshold():
    """
    Tests custom hysteresis threshold to replace that of scikit-image. This one
    only uses OpenCV and numpy methods.
    """
    # test array
    a = np.array([[6,6,7,4,2,3,1],
                [6,12,8,5,3,4,2],
                [6,6,7,4,2,7,1],
                [3,3,3,3,3,2,1],
                [2,2,2,2,2,2,2],
                [1,1,1,6,2,3,2],
                [1,1,1,1,1,1,2]], dtype='uint8')
    # known output
    a_desired = np.array([[255, 255, 255, 255,   0,   0,   0],
                           [255, 255, 255, 255,   0,   0,   0],
                           [255, 255, 255, 255,   0,   0,   0],
                           [  0,   0,   0,   0,   0,   0,   0],
                           [  0,   0,   0,   0,   0,   0,   0],
                           [  0,   0,   0,   0,   0,   0,   0],
                           [  0,   0,   0,   0,   0,   0,   0]], dtype='uint8')
    # specifies thresholds
    th_lo = 3
    th_hi = 7

    # performs hysteresis thresholding
    a_hyst = improc.hysteresis_threshold(a, th_lo, th_hi)
    # reports result
    passed = np.all(a_hyst == a_desired)

    return passed


def test_region_props(find_contours, input_filepath):
    """
    Compares speed of using findContours vs connectedComponentsWithStats from
    the OpenCV library to compute the region properties.
    """
    # loads video
    p = readin.load_params(input_filepath)

    # defines filepath to video
    vid_path = os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])

    # computes background with median filtering
    bkgd = improc.compute_bkgd_med_thread(vid_path, num_frames=p['num_frames_for_bkgd'])

    # measures the time
    ctr = 0
    time_total = 0
    # loops through frames of video
    for f in range(p['start'], p['end'], p['every']):
        # loads frame from video file
        frame, _ = basic.load_frame(vid_path, f)
        # extracts value channel of frame--including selem ruins segmentation
        val = basic.get_val_channel(frame)
        # subtracts reference image from current image (value channel)
        im_diff = cv2.absdiff(bkgd, val)

        ##################### THRESHOLD AND HIGH MIN SIZE #########################
        # thresholds image to become black-and-white
        thresh_bw = improc.thresh_im(im_diff, p['th'])
        # smooths out thresholded image
        closed_bw = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, p['selem'])
        # removes small objects
        bubble_bw = improc.remove_small_objects(closed_bw, p['min_size_th'])

        # fills enclosed holes with white, but leaves open holes black
        bubble_part_filled = basic.fill_holes(bubble_bw)
        # fills in holes that might be cut off at border
        frame_bw = improc.frame_and_fill(bubble_part_filled, p['width_border'])

        # measures time to compute region props
        time_start = time.time()
        if find_contours:
            bubbles = improc.region_props_find(frame_bw, ellipse=False)
        else:
            bubbles = improc.region_props_connected(frame_bw)

        time_total += (time.time() - time_start)
        ctr += 1

    return bubbles, ctr, time_total / float(ctr)




def test_remove_small_objects(find_contours, input_filepath):
    """
    Compares speed of different methods for removing small objects.
    """
    # loads video
    p = readin.load_params(input_filepath)

    # defines filepath to video
    vid_path = os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])

    # computes background with median filtering
    bkgd = improc.compute_bkgd_med(vid_path, num_frames=p['num_frames_for_bkgd'])

    # measures the time
    ctr = 0
    time_total = 0
    # loops through frames of video
    for f in range(p['start'], p['end'], p['every']):
        # loads frame from video file
        frame, _ = basic.load_frame(vid_path, f)
        # extracts value channel of frame--including selem ruins segmentation
        val = basic.get_val_channel(frame)
        # subtracts reference image from current image (value channel)
        im_diff = cv2.absdiff(bkgd, val)

        ##################### THRESHOLD AND HIGH MIN SIZE #########################
        # thresholds image to become black-and-white
        thresh_bw = improc.thresh_im(im_diff, p['th'])
        # smooths out thresholded image
        closed_bw = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, p['selem'])

        # measures time to remove small objects
        time_start = time.time()
        # removes small objects
        if find_contours:
            bubble_bw = improc.remove_small_objects(closed_bw, p['min_size_th'])
        else:
            bubble_bw = improc.remove_small_objects_connected(closed_bw, p['min_size_th'])

        time_total += (time.time() - time_start)
        ctr += 1

    return ctr, time_total / float(ctr)
