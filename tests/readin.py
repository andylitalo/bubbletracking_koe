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
import sys
sys.path.append('../src/')
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.basic as basic

# imports global conversions
from genl.conversions import *

# imports configurations and global variables
import config as cfg


############################## DEFINITIONS #####################################

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Track, label, and analyze bubbles in sheath flow.')
    ap.add_argument('tests', metavar='tests', type=int, nargs='+', 
                    help='tests to run (1-indexed)')
    ap.add_argument('-i', '--input_file', 
                    default='test/input_test.txt',
                    help='path to input file with remaining parameters')
    ap.add_argument('-c', '--check',
                    default=0, help='If 1, checks mask')
    ap.add_argument('-r', '--replace',
                    default=0, help='If 1, replaces existing files')
    args = vars(ap.parse_args())

    # extracts and formats individual parameters
    tests = list(args['tests'])
    input_file = args['input_file']
    check = int(args['check'])
    replace = int(args['replace'])

    return tests, input_file, check, replace


def load_params(input_file):
    """Loads and formats parameters in order from input file."""
    # reads all parameters from input file into a dictionary
    params = fn.read_input_file(input_file)

    # creates a dictionary to store processed input parameters
    p = {}

    # name of set of input parameters
    p['input_name'] = params['input_name']

    # image-processing parameters
    sd = int(params['selem_dim'])
    p['selem'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sd, sd))
    p['width_border'] = int(params['width_border'])
    p['start'] = int(params['start'])
    p['end'] = int(params['end'])
    p['every'] = int(params['every'])
    p['num_frames_for_bkgd'] = int(params['num_frames_for_bkgd'])
    p['th'] = int(params['th'])
    p['th_lo'] = int(params['th_lo'])
    p['th_hi'] = int(params['th_hi'])
    p['min_size_hyst'] = int(params['min_size_hyst'])
    p['min_size_th'] = int(params['min_size_th'])
    p['min_size_reg'] = int(params['min_size_reg'])

    # ensures that the number of frames for the background is less than total

    # file parameters
    p['vid_subdir'] = params['vid_subdir']
    p['vid_ext'] = params['vid_ext']

 
    return p