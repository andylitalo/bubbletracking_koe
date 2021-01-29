"""
lib.py contains functions for tests.

Author: Andy Ylitalo
Date: January 29, 2021
"""

import sys
sys.path.append('../src/')

import cv2

import cvimproc.improc as improc
import cvimproc.basic as basic
import genl.readin as readin


def test_remove_small_objects(find_contours, input_filepath, min_size):
    """
    Compares speed of different methods for removing small objects.
    """
    # loads video
    input_name, eta_i, eta_o, L, R_o, selem, width_border, fig_size_red, \
    num_frames_for_bkgd, start, end, every, th, th_lo, th_hi, \
    min_size_hyst, min_size_th, min_size_reg, highlight_method, \
    vid_subfolder, vid_name, expmt_folder, data_folder, fig_folder = readin.load_params(input_filepath)

    # defines filepath to video
    vid_path = expmt_folder + vid_subfolder + vid_name

    # computes background with median filtering
    bkgd = improc.compute_bkgd_med(vid_path, num_frames=num_frames_for_bkgd)

    # loops through frames of video
    for f in range(start, end, every):
        # loads frame from video file
        frame, _ = basic.load_frame(vid_path, f)
        # extracts value channel of frame--including selem ruins segmentation
        val = basic.get_val_channel(frame)
        # subtracts reference image from current image (value channel)
        im_diff = cv2.absdiff(bkgd, val)

        ##################### THRESHOLD AND HIGH MIN SIZE #########################
        # thresholds image to become black-and-white
        thresh_bw = improc.thresh_im(im_diff, th)
        # smooths out thresholded image
        closed_bw = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, selem)
        # removes small objects
        if find_contours:
            bubble_bw = improc.remove_small_objects_findContours(closed_bw, min_size_th)
        else:
            bubble_bw = improc.remove_small_objects(closed_bw, min_size_th)
