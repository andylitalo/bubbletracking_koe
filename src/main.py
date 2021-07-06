"""
main.py highlights, labels, and analyzes objects in videos

@author Andy Ylitalo
@date October 19, 2020
"""


# imports standard libraries
import os
import time

# imports custom libraries
# from libs
import cvimproc.improc as improc
import genl.readin as readin
import genl.main_helper as mh
import analysis.statlib as stat

# global variables and configurations
import config as cfg


def main():

    ####################### 0) PARSE INPUT ARGUMENTS ##########################
    input_file, check, print_freq, \
    replace, use_prev_bkgd, remember_objects = readin.parse_args()

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

    # TODO -- give option to provide mask that includes outer stream to
    # track outer-stream bubbles, too (e.g., for velocity mapping)

    # computes background (cropped)
    bkgd = mh.get_bkgd(vid_path, data_dir, row_lo, row_hi,
                    p['num_frames_for_bkgd'], use_prev_bkgd)

    # if thresholds not provided, estimates reasonable values (cf hist_pix.py`)
    if p['th'] == -1 or p['th_lo'] == -1 or p['th_hi'] == -1:
        th, th_lo, th_hi = stat.suggest_thresholds(vid_path, mask_data, bkgd, 
                                            p['start'], p['end'], p['every']) 
        # only updates thresholds that were given as -1
        p = mh.update_thresholds(p, th, th_lo, th_hi)

        print(p['th'], p['th_lo'], p['th_hi'])

    # computes flow properties
    dp, R_i, v_max, v_interf, \
    Q_i, Q_o, pix_per_um, d = mh.get_flow_props(vid_path, p)


    ######################## 2) TRACK OBJECTS ##################################
    # organizes arguments for object segmentation function
    object_kwargs = {'flow_dir' : flow_dir, 'pix_per_um' : pix_per_um, 
                        'R_i' : R_i, 'R_o' : p['R_o'], 'v_max' : v_max, 
                        'v_interf' : v_interf, 'd' : d, 'L' : p['L'], 
                        'dp' : dp}
    track_kwargs, highlight_kwargs, \
    assign_kwargs = mh.collect_kwargs(p, vid_path, bkgd, mask_data, flow_dir, 
                                    row_lo, row_hi, v_max, v_interf, pix_per_um,
                                    object_kwargs=object_kwargs, 
                                    remember_objects=remember_objects)
    # starts timer
    start_time = time.time()
    # tracks objects
    # TODO -- how do I add option to scale by bright field image?
    objs, frame_IDs = improc.track_obj(improc.track_obj_cvvidproc,
        track_kwargs, highlight_kwargs, assign_kwargs, ret_IDs=True)
    # stops timer and prints result
    print('{0:d} frames analyzed with object-tracking in {1:.3f} s.'.format(
                                        int((p['end']-p['start'])/p['every']),
					                    time.time()-start_time))

    # TODO Filter
    # TODO -- second round with greater sensitivity at specific locations
    # where bubbles are probable
    # TODO second filter --should these be separate methods?
    
    ######################## 3) PROCESS DATA ###################################
    for ID in objs.keys():
        obj = objs[ID]
        # computes average speed [m/s]
        obj.process_props()
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
