"""
track_bubbles.py highlights, labels, and analyzes bubbles in videos of bubble
nucleation and growth in sheath flow.

@author Andy Ylitalo
@date October 19, 2020
"""


# imports standard libraries
import pickle as pkl
import os
import shutil
import glob

# imports custom libraries
# from libs
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.mask as mask
import cvimproc.ui as ui
import genl.flow as flow
import cvimproc.basic as basic
import genl.readin as readin

# global conversions
from genl.conversions import *

# GLOBAL VARIABLES
input_folder = '../input/'


def main():

    ####################### 0) PARSE INPUT ARGUMENTS ###########################
    input_file, check, print_freq, replace, use_prev_bkgd = readin.parse_args()
    # determines filepath to input parameters (.txt file)
    input_path = input_folder + input_file

    ######################### 1) PRE-PROCESSING ################################

    # loads parameters
    params = readin.load_params(input_path)
    input_name, eta_i, eta_o, L, R_o, selem, width_border, \
    fig_size_red, num_frames_for_bkgd, \
    start, end, every, th, th_lo, th_hi, \
    min_size_hyst, min_size_th, min_size_reg, \
    highlight_method, \
    vid_subfolder, vid_name, \
    expmt_folder, data_folder, fig_folder = params

    # defines filepath to video
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
    pix_per_um = pix_per_um_dict[mag] # gets appropriate conversion for magnification

    # defines directory to video data and figures
    vid_dir = vid_subfolder + os.path.join(vid_name[:-4], input_name)
    data_dir = data_folder + vid_dir
    fig_dir = fig_folder + vid_dir
    # creates directories recursively if they do not exist
    fn.makedirs_safe(data_dir)
    fn.makedirs_safe(fig_dir)
    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(start, every, end))
    # if the data file already exists in the given directory, end analysis
    # upon request
    # if replacement requested, previous background may be used upon request
    if os.path.isfile(data_path):
        if not replace:
            print('{0:s} already exists. Terminating analysis.'.format(data_path))
            return

    # loads mask data; user creates new mask by clicking if none available
    first_frame, _ = basic.load_frame(vid_path, 0)
    flow_dir, mask_data = ui.click_sheath_flow(first_frame,
                                    vid_path[:-4]+'_mask.pkl', check=check)
    # computes minimum and maximum rows for bubble tracking computation
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)

    # gets background for image subtraction later on
    data_existing = glob.glob(os.path.join(data_dir, 'f_*.pkl'))
    if len(data_existing) > 0 and use_prev_bkgd:
        # loads background from first existing data file
        with open(data_existing[0], 'rb') as f:
            data_prev = pkl.load(f)
        bkgd = data_prev['metadata']['bkgd']
    else:
        # computes background with median filtering
        bkgd = improc.compute_bkgd_med_thread(vid_path,
            vid_is_grayscale=true,  #assume video is already grayscale (all RGB channels are the same)
            num_frames=num_frames_for_bkgd,
            crop_y=row_lo,
            crop_height=row_hi-row_lo)
        '''
        # old method
        bkgd = improc.compute_bkgd_med(vid_path, num_frames=num_frames_for_bkgd)
        '''

    # computes pressure drop [Pa], inner stream radius [m], and max velocity
    #  [m/s] for Poiseuille sheath flow
    dp, R_i, v_max = flow.get_dp_R_i_v_max(eta_i, eta_o, L, Q_i, Q_o, R_o,
                                            SI=True)

    ######################## 2) TRACK BUBBLES ##################################
    # organizes arguments for bubble segmentation function
    # TODO--how to let user customize list of arguments?
    args = (th, th_lo, th_hi, min_size_hyst, min_size_th, width_border, selem,
            mask_data)
    bubbles, frame_IDs = improc.track_bubble(vid_path, bkgd,
                                highlight_method,
                                args, pix_per_um, flow_dir, row_lo, row_hi,
                                v_max, min_size_reg=min_size_reg,
                                print_freq=print_freq,
                                width_border=width_border, ret_IDs=True,
                                start=start, end=end, every=every)

    ######################## 3) PROCESS DATA ###################################
    # computes velocity at interface of inner stream [m/s]
    v_inner = flow.v_inner(Q_i, Q_o, eta_i, eta_o, R_o, L)
    for ID in bubbles.keys():
        bubble = bubbles[ID]
        # computes average speed [m/s]
        bubble.proc_props()
        # compares average speed to cutoff [m/s]
        bubble.classify(v_inner)

    ########################## 4) SAVE RESULTS #################################
    # stores metadata (I will not store video parameters or parameters from the
    # input file since they are stored elsewhere already)
    metadata = {'input name' : input_name, 'bkgd' : bkgd, 'flow dir' : flow_dir,
                'mask data' : mask_data, 'row lo' : row_lo, 'row hi' : row_hi,
                'dp' : dp, 'R_i' : R_i, 'v_max' : v_max, 'v_inner' : v_inner,
                'args' : args, 'frame IDs' : frame_IDs,
                'pix_per_um' : pix_per_um, 'input params' : params}

    # stores data
    data = {}
    data['bubbles'] = bubbles
    data['frame IDs'] = frame_IDs
    data['metadata'] = metadata

    # also saves copy of input file and mask file
    shutil.copyfile(input_path,os.path.join(data_dir, input_file))
    shutil.copyfile(vid_path[:-4] + '_mask.pkl',
                        os.path.join(data_dir, 'mask.pkl'))

    # saves data
    with open(data_path, 'wb') as f:
        pkl.dump(data, f)

    return

if __name__ == '__main__':
    main()
