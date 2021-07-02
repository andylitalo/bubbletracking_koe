"""
highlight.py
@brief tests the success of highlighting bubbles by saving images of bubbles
highlighted using the current method. Great for quality control and trouble-
shooting good image-processing parameters.
@author Andy Ylitalo
@date October 24, 2020
"""

# adds libs folder to search path
import sys
sys.path.append('../src/')

# imports standard libraries
import os
import pickle as pkl
import cv2
import numpy as np
import argparse

# imports image-processing libraries
import skimage.measure
import skimage.color
import PIL.Image

# imports froms libs
import cvimproc.vid as vid
import cvimproc.basic as basic
import cvimproc.improc as improc
import cvimproc.mask as mask
import genl.fn as fn
import genl.main_helper as mh

sys.path.append('../analysis/')
import highlight

# local libraries
import readin

# imports configuration file
import config as cfg


def obj_label_color(obj, f, std_color=cfg.white, border_color=cfg.black):
    """
    Determines the color of the label to be printed on the image 
    for an object.

    Parameters
    ----------
    obj : TrackedObject
        Object to be labeled in image
    f : int
        Frame number in video
    std_color, border_color : 3-tuple of uint8s, optional
        Colors for labels of standard objects and objects on the border
        (default white and black)

    Returns
    -------
    color : 3-tuple of uint8s
        Color of label (RGB)
    """
    # text of number ID is black if on the border of the image, white o/w
    on_border = obj.get_prop('on border', f)
    if on_border:
        color = border_color
    else:
        color = std_color

    return color


def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Check quality of highlighting bubbles.')
    ap.add_argument('tests', metavar='tests', type=int, nargs='+',
                    help='list of tests to highlight objects in')
    ap.add_argument('-s', '--skip_blanks', default=1,
                    help='If 1, skips images without bubbles detected.')
    ap.add_argument('-b', '--brightness', default=3.0, type=float,
                    help='Factor to multiply image brightness by.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no apostrophes).')
    ap.add_argument('-i', '--input_file', default='test/input_test.txt',
                    help='Name of file with input parameters.')
    ap.add_argument('-c', '--color_object', default=1,
                    help='If 1, objects will be colored in figure.')
    ap.add_argument('-q', '--quiet', default=1, 
                        help='If 0, prints out each time image is saved')
    args = vars(ap.parse_args())

    return args

# TODO add option to save image with dimensions on tick marks

def main():
    # parses user-supplied information to identify data file for desired experiment
    args = parse_args()
    tests = list(args['tests'])
    skip_blanks = args['skip_blanks']
    brightness = args['brightness']
    ext = args['ext']
    input_file = args['input_file']
    color_object = args['color_object']
    quiet = args['quiet']

    # loads data file and parameters from mask and input.txt files
    input_path = os.path.join(cfg.input_dir, input_file)

    # highlights objects based on results of each requested test
    for i in tests:
        # refreshes parameters that changed
        p = readin.load_params(input_path)
        # gets paths to data, reporting failure if occurs
        p['vid_name'] = str(i) + p['vid_ext']

        # highlights and saves images in tracked video
        highlight.highlight_and_save_tracked_video(p, cfg.input_dir, cfg.output_dir, 
                            cfg.data_subdir, cfg.figs_subdir, 
                            skip_blanks=skip_blanks, brightness=brightness, 
                            ext=ext, color_object=color_object, quiet=quiet,
                            label_color_method=obj_label_color)

    return


if __name__ == '__main__':
    main()