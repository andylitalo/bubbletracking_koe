"""test_tracking.py

Performs tests on bubble-tracking algorithms to compare speed and accuracy of
koe's C++-based algorithms against my previously purely Python-based
algorithms.

Author: Andy Ylitalo
Date: March 2, 2021
"""

# imports local pkgs with implementation of image-processing
import sys
sys.path.append('../src/')
import cvimproc.improc as improc

# koe's library of C++-based algos for image-processing
import cv2
import cvvidproc


# loads input parameters
input_name, eta_i, eta_o, L, R_o, selem, width_border, fig_size_red, \
num_frames_for_bkgd, start, end, every, th, th_lo, th_hi, \
min_size_hyst, min_size_th, min_size_reg, _, \
vid_subfolder, vid_name, expmt_folder, data_folder, fig_folder = readin.load_params(input_filepath)

# defines parameters to compute background
vid_path = expmt_folder + vid_subfolder + vid_name

# checks that video has the requested frames
if not basic.check_frames(vid_path, end):
	print('Terminating analysis. Please enter valid frame range next time.')
	return

# extracts parameters recorded in video name
vid_params = fn.parse_vid_path(vid_path)
Q_i = uLmin_2_m3s*vid_params['Q_i'] # inner stream flow rate [m^3/s]
Q_o = uLmin_2_m3s*vid_params['Q_o'] # outer stream flow rate [m^3/s]
mag = vid_params['mag'] # magnification of objective (used for pix->um conv)
# TODO: create input method that allows user to specify since different from Photron
pix_per_um = pix_per_um_dict[mag] # gets appropriate conversion for magnification

### COMPUTES FLOW DIRECTION AND GENERATES MASK 	###
# loads mask data; user creates new mask by clicking if none available
first_frame, _ = basic.load_frame(vid_path, 0)
flow_dir, mask_data = ui.click_flow(first_frame,
			    vid_path[:-4]+'_mask.pkl', check=check)
# computes minimum and maximum rows for bubble tracking computation
row_lo, _, row_hi, _ = mask.get_bbox(mask_data)


# computes background
bkgd = improc.compute_bkgd_med_thread(vid_path, True, num_frames_for_bkgd)

# defines parameters to track bubbles
track_kwargs = {'bkgd' : bkgd,
		'vid_path' : vid_path,
		'start' : start,
		'end' : end,
		'every' : every}

highlight_kwargs = {'selem' : selem,
		'th' : th,
		'th_lo' : th_lo,
		'th_hi' : th_hi,
		'min_size_hyst' : min_size_hyst,
		'min_size_th' : min_size_th,
		'width_border' : width_border}

# parameters for assign_bubbles() method following ID_curr param
assignbubbles_kwargs = {'flow_dir' : flow_dir,
			'fps' : fps,
			'pix_per_um' : pix_per_um,
			'width_border' : width_border,
			'row_lo' : row_lo,
			'row_hi' : row_hi,
			'v_max' : v_max, 
			'min_size_reg' : min_size_reg}
		
# tracks bubble
bubbles_archive = improc.track_bubble_cvvidproc(track_kwargs,
				 highlight_kwargs, assignbubbles_kwargs)
