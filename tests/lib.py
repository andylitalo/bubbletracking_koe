"""
lib.py contains functions for tests.

Author: Andy Ylitalo
Date: January 29, 2021
"""

# imports standard libraries
import cv2
import time

# adds filepath to custom libraries
import sys
sys.path.append('../src/')
# imports custom libraries
import cvimproc.improc as improc
import cvimproc.basic as basic
import genl.readin as readin



def test_region_props(find_contours, input_filepath):
    """
    Compares speed of using findContours vs connectedComponentsWithStats from
    the OpenCV library to compute the region properties.
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

    # measures the time
    ctr = 0
    time_total = 0
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
        bubble_bw = improc.remove_small_objects(closed_bw, min_size_th)

        # fills enclosed holes with white, but leaves open holes black
        bubble_part_filled = improc.fill_holes(bubble_bw)
        # fills in holes that might be cut off at border
        frame_bw = improc.frame_and_fill(bubble_part_filled, width_border)

        # measures time to compute region props
        time_start = time.time()
        if find_contours:
            _ = improc.region_props_find(frame_bw, ellipse=False)
        else:
            _ = improc.region_props_connected(frame_bw)

        time_total += (time.time() - time_start)
        ctr += 1

    return ctr, time_total / float(ctr)




def test_remove_small_objects(find_contours, input_filepath):
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

    # measures the time
    ctr = 0
    time_total = 0
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

        # measures time to remove small objects
        time_start = time.time()
        # removes small objects
        if find_contours:
            bubble_bw = improc.remove_small_objects(closed_bw, min_size_th)
        else:
            bubble_bw = improc.remove_small_objects_connected(closed_bw, min_size_th)

        time_total += (time.time() - time_start)
        ctr += 1

    return ctr, time_total / float(ctr)
