"""
@package io.py
Handles input and output functions for track_bubbles.py.

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

# imports global conversions
from genl.conversions import *

# GLOBAL VARIABLES
highlight_methods = {'highlight_bubble_hyst_thresh' :
                            improc.highlight_bubble_hyst_thresh}


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
    # reads all parameters into a dictionary
    params = fn.read_input_file(input_file)

    # name of set of input parameters
    input_name = params['input_name']
    # flow parameters
    eta_i = float(params['eta_i'])
    eta_o = float(params['eta_o'])
    L = float(params['L'])
    R_o = um_2_m*float(params['R_o']) # [m]

    # image-processing parameters
    sd = int(params['selem_dim'])
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd, sd))
    width_border = int(params['width_border'])
    fig_size_red = float(params['fig_size_red'])
    num_frames_for_bkgd = int(params['num_frames_for_bkgd'])
    start_frame = int(params['start'])
    end_frame = int(params['end'])
    every = int(params['every'])
    th = int(params['th'])
    th_lo = int(params['th_lo'])
    th_hi = int(params['th_hi'])
    min_size_hyst = int(params['min_size_hyst'])
    min_size_th = int(params['min_size_th'])
    min_size_reg = int(params['min_size_reg'])
    h_m_str = params['highlight_method']
    assert h_m_str in highlight_methods, \
            '{0:s} not valid highlight method in readin.py'.format(h_m_str)
    highlight_method = highlight_methods[h_m_str]

    # file parameters
    vid_subfolder = params['vid_subfolder']
    vid_name = params['vid_name']
    expmt_folder = params['expmt_folder']
    data_folder = params['data_folder']
    fig_folder = params['fig_folder']

    return (input_name, eta_i, eta_o, L, R_o, selem, width_border, fig_size_red,
            num_frames_for_bkgd, start_frame, end_frame, every, th, th_lo, th_hi,
            min_size_hyst, min_size_th, min_size_reg, highlight_method,
            vid_subfolder, vid_name, expmt_folder, data_folder, fig_folder)
