"""
basic_tracking.py highlights, labels, and analyzes bubbles in test videos
of white circles moving across a black background.

Modify from `main.py` from `src` folder.

@author Andy Ylitalo
@date May 3, 2021
"""


# imports standard libraries
import os
import shutil
import time

# imports custom libraries
# from libs
import sys
sys.path.append('../src/')
import genl.fn as fn
import cvimproc.improc as improc
import genl.main_helper as mh
import cvimproc.basic as basic
from classes.classes import TrackedObject
# local file
import readin

# global variables and configurations
import config as cfg


def main():

    ####################### 0) PARSE INPUT ARGUMENTS ###########################
    tests, input_file, check, replace = readin.parse_args()
    ######################### 1) PRE-PROCESSING ################################

    # determines filepath to input parameters (.txt file)
    input_path = os.path.join(cfg.input_dir, input_file)


    # performs analysis for each video in list of tests
    for i in tests:

        # loads parameters (each time to refresh changed parameters)
        p = readin.load_params(input_path)

        # gets paths to data, reporting failure if occurs
        p['vid_name'] = str(i) + p['vid_ext']
        # if last frame given as -1, returns as final frame of video
        # *Note: must move up one directory to `src` for correct filepath
        vid_path =  os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
        p['end'] = basic.get_frame_count(vid_path, p['end'])
        # ensures that the number of frames for the background is less than total
        p['num_frames_for_bkgd'] = min(int(p['num_frames_for_bkgd']), p['end'])
        
        _, data_path, vid_dir, \
        data_dir, figs_dir, stop = mh.get_paths(p, replace)
        if stop:
            return

        p['num_frames_for_bkgd'] = min(int(p['num_frames_for_bkgd']), p['end'])

            
         # gets mask and related parameters for video (user interaction)
        mask_data, flow_dir, row_lo, row_hi = mh.get_mask(vid_path, check=check)

        # computes background
        bkgd = mh.get_bkgd(vid_path, data_dir, row_lo, row_hi,
                    p['num_frames_for_bkgd'], False)


        ######################## 2) TRACK BUBBLES ##################################
        # organizes arguments for object segmentation function
        # object-tracking keyword arguments
        track_kwargs = {'vid_path' : vid_path,
            'bkgd' : bkgd,
            'assign_objects_method' : cfg.assign_method,
            'highlight_method' : cfg.highlight_method, # only used for pure-Python analysis
            'start' : p['start'],
            'end' : p['end'],
            'every' : p['every']
        }

        # object highlighting/segmentation keyword arguments
        highlight_kwargs = {'th' : p['th'],
            'th_lo' : p['th_lo'],
            'th_hi' : p['th_hi'],
            'min_size_hyst' : p['min_size_hyst'],
            'min_size_th' : p['min_size_th'],
            'width_border' : p['width_border'],
            'selem' : p['selem'],
            'mask_data' : mask_data,
            'only_dark_obj' : False,
        }

        # additional variables required for method to measure distance between objects
        d_fn_kwargs = {
            'flow_dir' : flow_dir
        }
        # label assignment/tracking keyword arguments
        assign_kwargs = {
            'fps' : 1,
            'd_fn' : improc.d_off_flow,
            'd_fn_kwargs' : d_fn_kwargs,
            'width_border' : p['width_border'],
            'min_size_reg' : p['min_size_reg'],
            'row_lo' : row_lo,
            'row_hi' : row_hi,
            'remember_objects' : True,
            'ellipse' : True,
            'ObjectClass' : TrackedObject,
            'object_kwargs' : {},
        }               
        
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
            obj.process_props()

        ########################## 4) SAVE RESULTS #################################
        # stores metadata (I will not store video parameters or parameters from the
        # input file since they are stored elsewhere already)
        mh.save_data(objs, frame_IDs, p, track_kwargs, highlight_kwargs, assign_kwargs, 
                    vid_path, input_path, data_path)

    return

if __name__ == '__main__':
    main()
