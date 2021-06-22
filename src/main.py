"""
track_objs.py highlights, labels, and analyzes objects in videos

@author Andy Ylitalo
@date October 19, 2020

@bug June 15, 2021: GStreamer fails when trying to use "VideoCapture" feature of cv2
"""


# imports standard libraries
import pickle as pkl
import os
import shutil
import glob
import time

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
# global variables and configurations
import config as cfg

# GLOBAL VARIABLES TODO -- make these inputs
d_fn = improc.bubble_distance_v



def main():

    ####################### 0) PARSE INPUT ARGUMENTS ###########################
    input_file, check, print_freq, replace, use_prev_bkgd = readin.parse_args()
    # determines filepath to input parameters (.txt file)
    input_path = os.path.join(cfg.input_dir, input_file)


    ######################### 1) PRE-PROCESSING ################################

    # loads parameters
    p = readin.load_params(input_path)

    # defines filepath to video
    vid_path = os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])

    # checks that video has the requested frames
    # subtracts 1 since "end" gives an exclusive upper bound [start, end)
    if not basic.check_frames(vid_path, p['end']-1):
        print('Terminating analysis. Please enter valid frame range next time.')
        return

    # extracts parameters recorded in video name
    vid_params = fn.parse_vid_path(vid_path)
    Q_i = uLmin_2_m3s*vid_params['Q_i'] # inner stream flow rate [m^3/s]
    Q_o = uLmin_2_m3s*vid_params['Q_o'] # outer stream flow rate [m^3/s]
    mag = vid_params['mag'] # magnification of objective (used for pix->um conv)
    # gets unit conversion for Photron camera based on microscope magnification
    if p['photron']:
        pix_per_um = pix_per_um_photron[mag]
    # otherwise uses conversion for Chronos camera
    else:
        pix_per_um = pix_per_um_chronos[mag]

    # defines directory to video data and figures
    vid_dir = os.path.join(p['vid_subdir'], p['vid_name'][:-4], p['input_name'])
    data_dir = os.path.join(cfg.output_dir, vid_dir, cfg.data_subdir)
    figs_dir = os.path.join(cfg.output_dir, vid_dir, cfg.figs_subdir)

    # creates directories recursively if they do not exist
    fn.makedirs_safe(data_dir)
    fn.makedirs_safe(figs_dir)

    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'],
                                                        p['every'], p['end']))
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
    # computes minimum and maximum rows for object tracking computation
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
            vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
            num_frames=p['num_frames_for_bkgd'],
            crop_y=row_lo,
            crop_height=row_hi-row_lo)

    # computes pressure drop [Pa], inner stream radius [m], and max velocity
    #  [m/s] for Poiseuille sheath flow
    dp, R_i, v_max = flow.get_dp_R_i_v_max(p['eta_i'], p['eta_o'], p['L'],
                                        Q_i, Q_o, p['R_o'], SI=True)

    ######################## 2) TRACK OBJECTS ##################################
    # organizes arguments for object segmentation function
    # TODO--how to let user customize list of arguments?
    track_kwargs = {'vid_path' : vid_path,
        'bkgd' : bkgd,
        'assign_objects_method' : improc.assign_objects,
        'highlight_method' : p['highlight_method'], # only used for pure-Python analysis
        'print_freq' : print_freq,
        'start' : p['start'],
        'end' : p['end'],
        'every' : p['every']
    }

    highlight_kwargs = {'th' : p['th'],
        'th_lo' : p['th_lo'],
        'th_hi' : p['th_hi'],
        'min_size_hyst' : p['min_size_hyst'],
        'min_size_th' : p['min_size_th'],
        'width_border' : p['width_border'],
        'selem' : p['selem'],
        'mask_data' : mask_data
    }

    # additional variables required for method to measure distance between objects
    # TODO how do I make this assignment more general and less function-specific?
    d_fn_kwargs = {
        'axis' : flow_dir,
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'v_max' : v_max*m_2_um*pix_per_um,  # convert max velocity from [m/s] to [pix/s] first
        'fps' : fn.parse_vid_path(vid_path)['fps'],
    }

    # additional variables required for method used to assign labels to objects
    assign_kwargs = {
        'fps' : fn.parse_vid_path(vid_path)['fps'],  # extracts fps from video filepath
        'd_fn' : d_fn,
        'd_fn_kwargs' : d_fn_kwargs,
        'width_border' : p['width_border'],
        'min_size_reg' : p['min_size_reg'],
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'remember_objects' : False, # TODO -- make this a parameter that can be changed by user
        'object_kwargs' : {'flow_dir' : flow_dir, 'pix_per_um' : pix_per_um},
    }               

    ### OBJECT TRACKING ###
    # starts timer
    start_time = time.time()
    # tracks objects
    objs, frame_IDs = improc.track_obj(improc.track_obj_cvvidproc,
        track_kwargs, highlight_kwargs, assign_kwargs, ret_IDs=True)
    # stops timer and prints result
    print('{0:d} frames analyzed with object-tracking in {1:.3f} s.'.format(
                                        int((p['end']-p['start'])/p['every']),
					time.time()-start_time))

    ######################## 3) PROCESS DATA ###################################
    # computes velocity at interface of inner stream [m/s]
    v_inner = flow.v_inner(Q_i, Q_o, p['eta_i'], p['eta_o'], p['R_o'], p['L'])
    for ID in objs.keys():
        obj = objs[ID]
        # computes average speed [m/s]
        obj.proc_props()

    ########################## 4) SAVE RESULTS #################################
    # stores metadata (I will not store video parameters or parameters from the
    # input file since they are stored elsewhere already)
    metadata = {'input name' : p['input_name'], 'bkgd' : bkgd, 'flow dir' : flow_dir,
                'mask data' : mask_data, 'row lo' : row_lo, 'row hi' : row_hi,
                'dp' : dp, 'R_i' : R_i, 'v_max' : v_max, 'v_inner' : v_inner,
                'args' : highlight_kwargs, 'frame IDs' : frame_IDs,
                'pix_per_um' : pix_per_um, 'input params' : p}

    # stores data
    data = {}
    data['objects'] = objs
    data['frame IDs'] = frame_IDs
    data['metadata'] = metadata

    # also saves copy of input file and mask file
    shutil.copyfile(input_path, os.path.join(data_dir, input_file))
    shutil.copyfile(vid_path[:-4] + '_mask.pkl',
                        os.path.join(data_dir, 'mask.pkl'))

    # saves data
    with open(data_path, 'wb') as f:
        pkl.dump(data, f)

    return

if __name__ == '__main__':
    main()
