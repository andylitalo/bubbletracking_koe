"""
@package readin.py
Handles input and output functions for main.py.

@author Andy Ylitalo
@date October 19, 2020
"""


# imports system libraries
import argparse
import shutil
import os

# imports image-processing libraries
import cv2

# imports custom libraries
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.basic as basic

# imports global conversions
import genl.conversions as conv

# imports configurations and global variables
import sys
sys.path.append('../')
import config as cfg


############################## DEFINITIONS #####################################

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Track, label, and analyze objects in sheath flow.')
    ap.add_argument('-i', '--input_file', default='input.txt',
                    help='path to input file with remaining parameters')
    ap.add_argument('-c', '--check_mask', default=1, type=int,
                    help='1 to check mask, 0 to accept if available')
    ap.add_argument('-f', '--freq', default=10, type=int,
                    help='frequency of printouts while processing frames.')
    ap.add_argument('-r', '--replace', default=0, type=int,
                    help='If 1, replaces existing data file if present.' + \
                    ' Otherwise, if 0, does not replace data file if present.')
    ap.add_argument('-b', '--use_prev_bkgd', default=0, type=int,
                    help='If 1, uses previous background if data available.')
    ap.add_argument('-m', '--remember', default=0, type=int,
                    help='If 1, remembers objects by extrapolating centroid.')
    args = vars(ap.parse_args())

    # extracts and formats individual parameters
    input_file = args['input_file']
    check = bool(args['check_mask'])
    print_freq = args['freq']
    replace = bool(args['replace'])
    use_prev_bkgd = bool(args['use_prev_bkgd'])
    remember_objects = bool(args['remember'])

    return input_file, check, print_freq, replace, use_prev_bkgd, remember_objects


def load_params(input_file):
    """Loads and formats parameters in order from input file."""
    # reads all parameters from input file into a dictionary
    params = fn.read_input_file(input_file)

    # creates a dictionary to store processed input parameters
    p = {}

    # name of set of input parameters
    p['input_name'] = params['input_name']
    # flow parameters
    p['eta_i'] = float(params['eta_i'])
    p['eta_o'] = float(params['eta_o'])
    p['L'] = float(params['L'])
    p['R_o'] = conv.um_2_m*float(params['R_o']) # [m]

    # image-processing parameters
    sd = int(params['selem_dim'])
    p['selem'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd, sd))
    p['width_border'] = int(params['width_border'])
    p['start'] = int(params['start'])
    p['end'] = int(params['end'])
    p['every'] = int(params['every'])
    # determines thresholds
    p['th'] = int(params['th']) # uniform threshold
    p['th_lo'] = int(params['th_lo']) # low hysteresis threshold
    p['th_hi'] = int(params['th_hi']) # high hysteresis threshold        

    # makes sure thresholds are properly ordered (prevents input errors)
    assert correct_thresholds(p), \
            'thresholds not properly ordered th_lo < th < th_hi'
    p['min_size_hyst'] = int(params['min_size_hyst'])
    p['min_size_th'] = int(params['min_size_th'])
    p['min_size_reg'] = int(params['min_size_reg'])
    p['camera'] = params['camera']

    # file parameters
    p['vid_subdir'] = params['vid_subdir']
    p['vid_name'] = params['vid_name']

    # if last frame given as -1, returns as final frame of video
    # *Note: must move up one directory to `src` for correct filepath
    vid_path =  os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
    p['end'] = basic.get_frame_count(vid_path, p['end'])

    # ensures that the number of frames for the background is less than total
    p['num_frames_for_bkgd'] = min(int(params['num_frames_for_bkgd']), p['end'])
    
    return p
    

def correct_thresholds(p):
    """
    Checks that the thresholds are ordered th_lo < th < th_hi
    """
    return ( (p['th_lo'] < p['th']) or p['th'] == -1 ) and \
            ( (p['th'] < p['th_hi']) or p['th_hi'] == -1 ) 