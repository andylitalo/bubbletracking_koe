"""
basic_tracking.py highlights, labels, and analyzes bubbles in test videos
of white circles moving across a black background.

Based on `main.py` from `src` folder.

@author Andy Ylitalo
@date May 3, 2021
"""


# imports standard libraries
import pickle as pkl
import os
import shutil
import glob
import time

# imports custom libraries
# from libs
import sys
sys.path.append('../src/')
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.mask as mask
import cvimproc.ui as ui
import genl.flow as flow
import cvimproc.basic as basic
# local file
import readin

# global conversions
from genl.conversions import *
# global variables and configurations
import config as cfg


def main():

    ####################### 0) PARSE INPUT ARGUMENTS ###########################
    tests, input_file, check, replace = readin.parse_args()
    # determines filepath to input parameters (.txt file)
    input_path = cfg.input_dir + input_file

    ######################### 1) PRE-PROCESSING ################################

    # loads parameters
    p = readin.load_params(input_path)


    # performs analysis for each video in list of tests
    for i in tests:


        # defines filepath to video
        vid_name = str(i) + p['vid_ext']
        vid_path = cfg.input_dir + p['vid_subdir'] + vid_name

        # checks that video has the requested frames
        # subtracts 1 since "end" gives an exclusive upper bound [start, end)
        
        # counts frames
        if p['end'] == -1:
            end = basic.count_frames(vid_path) 
        else:
            end = p['end']
        if not basic.check_frames(vid_path, end-1):
            print('Terminating analysis. Please enter valid frame range next time.')
            continue
        
        # defines directory to video data and figures (removes ext from video name)
        vid_dir = p['vid_subdir'] + os.path.join(vid_name[:-4], p['input_name'])
        data_dir = cfg.output_dir + vid_dir + cfg.data_subdir
        figs_dir = cfg.output_dir + vid_dir + cfg.figs_subdir
        # creates directories recursively if they do not exist
        fn.makedirs_safe(data_dir)
        fn.makedirs_safe(figs_dir)
        # defines name of data file to save
        data_path = os.path.join(data_dir,
                                'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'], 
                                                            p['every'], end))
        # if the data file already exists in the given directory, end analysis
        # upon request
        # if replacement requested, previous background may be used upon request
        if os.path.isfile(data_path):
            if not replace:
                print('{0:s} already exists. Terminating analysis.'.format(data_path))
                continue 
        # loads mask data; user creates new mask by clicking if none available
        first_frame, _ = basic.load_frame(vid_path, 0)
        flow_dir, mask_data = ui.click_sheath_flow(first_frame,
                                        vid_path[:-4]+'_mask.pkl', check=check)
        # computes minimum and maximum rows for bubble tracking computation
        row_lo, _, row_hi, _ = mask.get_bbox(mask_data)

        # computes background with median filtering
        bkgd = improc.compute_bkgd_med_thread(vid_path,
            vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
            num_frames=p['num_frames_for_bkgd'],
            crop_y=row_lo,
            crop_height=row_hi-row_lo)
        '''
        # old method
        bkgd = improc.compute_bkgd_med(vid_path, num_frames=num_frames_for_bkgd)
        '''


        ######################## 2) TRACK BUBBLES ##################################
        # organizes arguments for bubble segmentation function
        # TODO--how to let user customize list of arguments?
        track_kwargs = {'vid_path' : vid_path,
            'bkgd' : bkgd,
            'highlight_bubble_method' : p['highlight_method'],
            'print_freq' : 10,
            'start' : p['start'],
            'end' : end,
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

        assignbubbles_kwargs = {'pix_per_um' : 1,
            'flow_dir' : flow_dir,
            'fps' : 1,
            'row_lo' : row_lo,
            'row_hi' : row_hi,
            'v_max' : 100,
            'min_size_reg' : p['min_size_reg'],
            'width_border' : p['width_border']
        }

        start_time = time.time()
        bubbles, frame_IDs = improc.track_bubble(improc.track_bubble_cvvidproc,
            track_kwargs, highlight_kwargs, assignbubbles_kwargs, ret_IDs=True)
        print('{0:d} frames analyzed with bubble-tracking in {1:.3f} s.'.format(
                                            int((end-p['start'])/p['every']),
                                            time.time()-start_time)) 

        ######################## 3) PROCESS DATA ###################################
        for ID in bubbles.keys():
            bubble = bubbles[ID]
            # computes average speed [m/s]
            bubble.proc_props()

        ########################## 4) SAVE RESULTS #################################
        # stores metadata (I will not store video parameters or parameters from the
        # input file since they are stored elsewhere already)
        metadata = {'input name' : p['input_name'], 'bkgd' : bkgd,
                    'mask data' : mask_data, 'row lo' : row_lo, 'row hi' : row_hi,
                    'args' : highlight_kwargs, 'frame IDs' : frame_IDs,
                    'input params' : p}

        # stores data
        data = {}
        data['bubbles'] = bubbles
        data['frame IDs'] = frame_IDs
        data['metadata'] = metadata

        # also saves copy of input file and mask file
        shutil.copyfile(input_path, os.path.join(data_dir, 'input.txt'))
        shutil.copyfile(vid_path[:-4] + '_mask.pkl',
                            os.path.join(data_dir, 'mask.pkl'))

        # saves data
        with open(data_path, 'wb') as f:
            pkl.dump(data, f)

    return

if __name__ == '__main__':
    main()
