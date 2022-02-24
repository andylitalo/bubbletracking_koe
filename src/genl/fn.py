# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:33:41 2020

@author: Andy
fn.py contains useful, short, and often-used functions.
"""

import os
import cv2
import numpy as np
import re


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
    Checks if the version of OpenCV is 3.X.X or higher.
    The keywords for accessing properties from an OpenCV
    VideoCapture object are different after v2.X.X.
    """
    (major, minor, _) = cv2.__version__.split('.')
    return int(major) >= 3


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
    """
    Parses video folder of format <yyyymmdd>_<p_sat><units>
    
    Parameters
    ----------
    vid_folder : string
        Directory in which video is saved (excludes filename)
    
    Returns
    -------
    date : string
        Date of experiment <yyyymmdd>
    p_sat : int
        Pressure at which inner stream was saturated 
    p_sat_units : string
        Units of p_sat
    """
    result = re.search('20[0-9]{6}_[0-9]{2,}[a-zA-z]{3,}', vid_folder)
    if result is None:
        return None, None, None
    else:
        vid_str = result.group()
        date = re.search('^20[0-9]{6}', vid_str).group()
        p_sat = int(re.search('_[0-9]{2,}', vid_str).group()[1:])
        units = re.search('[a-zA-z]{3,}', vid_str).group()

        return date, p_sat, units


def parse_vid_path(vid_path):
    """
    Parses the video filepath to extract metadata.
        
    Parameters
    ----------
    vid_path : string
        Filepath to video of form <directory>/
        <prefix>_<fps>_<exposure time>_<inner stream flow rate [uL/min]>_
        <outer stream flow rate [uL/min]>_<distance along observation capillary [mm]>_
        <magnification of objective lens>_<number of video in series>.<ext>
    
    Returns
    -------
    params : dictionary
        Items: 'prefix' (string, usually polyol and gas), 
            'fps' (int, frames per second), 'exp_time' (float, exposure time [us]),
            'Q_i' (float, inner stream flow rate [ul/min]), 
            'Q_o' (int, outer stream flow rate [uL/min]), 
            'd' (int, distance along observation capillary [mm]),
            'mag' (int, magnification of objective lens), 'num' (int, number of video in series)
    """
    i_start = vid_path.rfind(os.path.sep)
    vid_file = vid_path[i_start+1:]
    # cuts out extension and splits by underscores
    match = re.search('_[0-9]{4,5}_[0-9]{3}(-[0-9]{1})?_[0-9]{3}_[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]+', vid_file)
    span = match.span()
    prefix = vid_file[:span[0]]
    param_str = match.group()
    tokens = param_str.split('_')

    params = {'prefix' : prefix,
              'fps' : int(tokens[1]),
              'exp_time' : read_dash_decimal(tokens[2]),
              'Q_i' : read_dash_decimal(tokens[3]),
              'Q_o' : int(tokens[4]),
              'd' : int(tokens[5]),
              'mag' : int(tokens[6]),
              'num' : int(tokens[7])}

    return params


def read_dash_decimal(num_str):
    """Reads string as float where dash '-' is used as decimal point."""
    result = 0
    if '-' in num_str:
        val, dec = num_str.split('-')
        result = int(val) + int(dec)/10.0**(len(dec))
    else:
        result = int(num_str)

    return result


def read_input_file(input_path, split_char='=', cmnt_char='#'):
    """
    Loads parameters from input file for tracking objects.

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
    folder_list_padded = path.split(os.path.sep)
    folder_list = [folder for folder in folder_list_padded if folder != '']

    return folder_list
