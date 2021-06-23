"""
track_objs.py highlights, labels, and analyzes objects in videos

@author Andy Ylitalo
@date October 19, 2020

@bug June 15, 2021: GStreamer fails when trying to use "VideoCapture" feature of cv2
"""


# imports standard libraries
import os
import time

# imports custom libraries
# from libs
import cvimproc.improc as improc
import genl.readin as readin
import genl.main_helper as mh

# global variables and configurations
import config as cfg


def main():

    ####################### 0) PARSE INPUT ARGUMENTS ##########################
    input_file, check, print_freq, replace, use_prev_bkgd = readin.parse_args()

    ######################### 1) PRE-PROCESSING ################################
    # loads parameters
    input_path = os.path.join(cfg.input_dir, input_file)
    p = readin.load_params(input_path)

    # gets paths to data, reporting failure if occurs
    vid_path, data_path, vid_dir, \
    data_dir, figs_dir, stop = mh.get_paths(p, replace)
    if stop:
        return

    # gets mask and related parameters for video (user interaction)
    mask_data, flow_dir, row_lo, row_hi = mh.get_mask(vid_path, check=check)

    # computes background
    bkgd = mh.get_bkgd(vid_path, data_dir, row_lo, row_hi,
                    p['num_frames_for_bkgd'], use_prev_bkgd)

    # computes flow properties
    dp, R_i, v_max, v_interf, Q_i, Q_o, pix_per_um = mh.get_flow_props(vid_path, p)

    ######################## 2) TRACK OBJECTS ##################################
    # organizes arguments for object segmentation function
    object_kwargs = {'flow_dir' : flow_dir, 'pix_per_um' : pix_per_um}
    track_kwargs, highlight_kwargs, \
    assign_kwargs = mh.collect_kwargs(p, vid_path, bkgd, mask_data, flow_dir, 
                                    row_lo, row_hi, v_max, v_interf, pix_per_um,
                                    object_kwargs=object_kwargs)
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
    for ID in objs.keys():
        obj = objs[ID]
        # computes average speed [m/s]
        obj.proc_props()
        # classifies object as inner or outer stream (or error)
        obj.classify(v_interf)

    ########################## 4) SAVE RESULTS #################################
    # stores metadata (I will not store video parameters or parameters from the
    # input file since they are stored elsewhere already)
    mh.save_data(objs, frame_IDs, p, track_kwargs, highlight_kwargs, assign_kwargs, 
                vid_path, input_path, data_path)

    return

if __name__ == '__main__':
    main()
