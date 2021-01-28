# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:33:41 2020

@author: Andy
fn.py contains useful, short, and often-used functions.
"""

import os
import cv2
import numpy as np


def bool_2_uint8(bool_arr):
    """
    Converts a boolean array to a black-and-white
    uint8 array (0 and 255).
    PARAMETERS:
        im : (M x N) numpy array of bools
            boolean array to convert
    RETURNS:
        (result) : (M x N) numpy array of uint8s
            uint8 array of 0s (False) and 255s (True)
    """
    assert (bool_arr.dtype == 'bool'), \
        'improc.bool_2_uint8() only accepts boolean arrays.'
    return (255*bool_arr).astype('uint8')


def format_float(i):
    """Formats string representation of float using "-" as decimal point."""
    result = 0
    if '-' in i:
        val, dec = i.split('-')
        result = int(val) + int(dec)/10.0**(len(dec))
    else:
        result = int(i)

    return result


def get_fps(vid_path, prefix):
    """
    Gets the frames per second from the path of the video.

    Parameters
    ----------
    vid_path : string
        path to video with frames per second in characters following prefix.
    prefix : string
        prefix given to all videos before their specs, e.g., 'v360_co2_'
    Returns
    -------
    fps : int
        frames per second of video.

    """
    _, name = os.path.split(vid_path)
    i1 = name.find(prefix) + len(prefix)
    i2 = name[i1:].find('_')
    fps = int(name[i1:i1+i2])

    return fps

def is_cv3():
    """
    Checks if the version of OpenCV is cv3.
    """
    (major, minor, _) = cv2.__version__.split('.')
    return int(major) == 3


def makedirs_safe(path):
    """os.makedirs() but checks if exists first."""
    if not os.path.isdir(path):
        os.makedirs(path)


def one_2_uint8(one_arr):
    """
    Converts an array of floats scaled from 0-to-1 to a
    uint8 array (0 and 255).
    PARAMETERS:
        im : (M x N) numpy array of floats
            floats from 0 to 1 to convert
    RETURNS:
        (result) : (M x N) numpy array of uint8s
            uint8 array from 0 to 255
    """
    assert (one_arr.dtype == 'float' and np.max(one_arr <= 1.0)), \
        'improc.one_2_uint8() only accepts floats arrays from 0 to 1.'
    return (255*one_arr).astype('uint8')


def parse_vid_folder(vid_folder):
    """Parses video folder of format <yyyymmdd>_<p_sat>bar"""
    # first extracts direct folder for video if others included in path
    vid_folder_list = split_folders(vid_folder)
    vid_folder = vid_folder_list[-1]
    raw = vid_folder.strip(os.path.sep)
    date, p_sat_str = raw.split('_')
    p_sat = int(''.join([c for c in p_sat_str if c.isdigit()]))
    p_sat_units = ''.join([c for c in p_sat_str if not c.isdigit()])

    return date, p_sat, p_sat_units

def parse_vid_path(vid_path):
    i_start = vid_path.rfind(os.path.sep)
    vid_file = vid_path[i_start+1:]
    # cuts out extension and splits by underscores
    tokens = vid_file[:-4].split('_')
    prefix = ''

    for i, token in enumerate(tokens):
        if token.isnumeric():
            break
        prefix.join(token)

    params = {'prefix' : prefix,
              'fps' : int(tokens[i]),
              'exp_time' : int(tokens[i+1]),
              'Q_i' : format_float(tokens[i+2]),
              'Q_o' : int(tokens[i+3]),
              'd' : int(tokens[i+4]),
              'P' : int(tokens[i+5]),
              'mag' : int(tokens[i+6]),
              'num' : int(tokens[i+7])}

    return params


def read_input_file(input_path, split_char='=', cmnt_char='#'):
    """
    Loads parameters from input file for tracking bubbles.

    Assumes file format of
    <param> <split_char> <value> <cmnt_char> <comment>

    where <split_char> is the character defining the split between parameter
    name and value and <cmnt_char> marks comments. Lines without this structure
    are ignored.

    Parameters
    ----------
    input_file : string
        Path to file from which parameters are loaded.
    split_char : char, optional
        Character defining split between parameter name and value.
        Default is '='.
    cmnt_char : char, optional
        Character marking an inline comment. Default is '#'.

    Returns
    -------
    params : dictionary
        Dictionary with parameter names as keys and parameter values as values.

    """
    # initializes dictionary to hold parameters
    params = {}

    # opens file
    file_obj = open(input_path)

    # reads in parameters from the lines of the file
    for line in file_obj:
        # removes comments at the end of the line
        line = line.split(cmnt_char, 1)[0]
        # removes spaces padding the line
        line = line.strip()
        key_value = line.split(split_char)
        # checks that there is a key and value (2 items)
        if len(key_value) == 2:
            # loads parameter
            params[key_value[0].strip()] = key_value[1].strip()

    return params


def remove_nans(arr):
    """
    Removes nans from array and returns indices of elements that are not nans.
    """
    not_nan = [i for i in range(len(arr)) if not np.isnan(arr[i])]

    return not_nan, arr[not_nan]


def split_folders(path):
    """Suggested on http://nicks-liquid-soapbox.blogspot.com/2011/03/
    splitting-path-to-list-in-python.html"""
    path_norm = os.path.normpath(path)
    folder_list_padded = path.split(os.path.sep)
    folder_list = [folder for folder in folder_list_padded if folder != '']

    return folder_list
