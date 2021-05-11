"""
@package readin.py
Handles input and output functions for main.py.

@author Andy Ylitalo
@date October 19, 2020
"""


# imports system libraries
import argparse
import shutil

# imports image-processing libraries
import cv2

# imports custom libraries
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.basic as basic

# imports global conversions
from genl.conversions import *

# imports configurations and global variables
import sys
sys.path.append('../')
import config as cfg


############################## DEFINITIONS #####################################

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Track, label, and analyze bubbles in sheath flow.')
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
    args = vars(ap.parse_args())

    # extracts and formats individual parameters
    input_file = args['input_file']
    check = bool(args['check_mask'])
    print_freq = args['freq']
    replace = bool(args['replace'])
    use_prev_bkgd = bool(args['use_prev_bkgd'])

    return input_file, check, print_freq, replace, use_prev_bkgd


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
    p['R_o'] = um_2_m*float(params['R_o']) # [m]

    # image-processing parameters
    sd = int(params['selem_dim'])
    p['selem'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd, sd))
    p['width_border'] = int(params['width_border'])
    p['num_frames_for_bkgd'] = int(params['num_frames_for_bkgd'])
    p['start'] = int(params['start'])
    p['end'] = int(params['end'])
    p['every'] = int(params['every'])
    p['th'] = int(params['th'])
    p['th_lo'] = int(params['th_lo'])
    p['th_hi'] = int(params['th_hi'])
    p['min_size_hyst'] = int(params['min_size_hyst'])
    p['min_size_th'] = int(params['min_size_th'])
    p['min_size_reg'] = int(params['min_size_reg'])
    p['photron'] = bool(params['photron'])
    h_m_str = params['highlight_method']
    assert h_m_str in cfg.highlight_methods, \
            '{0:s} not valid highlight method in readin.py'.format(h_m_str)
    p['highlight_method'] = cfg.highlight_methods[h_m_str]

    # file parameters
    p['vid_subdir'] = params['vid_subdir']
    p['vid_name'] = params['vid_name']

    # if last frame given as -1, returns as final frame of video
    if p['end'] == -1:
        p['end'] = basic.count_frames(cfg.input_dir + p['vid_subdir'] + \
                                                         p['vid_name'])
        
    return p

    #return (input_name, eta_i, eta_o, L, R_o, selem, width_border, fig_size_red,
    #        num_frames_for_bkgd, start, end, every, th, th_lo, th_hi,
    #        min_size_hyst, min_size_th, min_size_reg, highlight_method,
    #        vid_subdir, vid_name, expmt_dir)