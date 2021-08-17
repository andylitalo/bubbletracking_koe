"""
intensity_profiles.py plots intensity profiles of images of objects
after background subtraction next to the original image to give a 
more tangible sense of edge gradients and signal-to-noise ratio.

Author: Andy Ylitalo
Date: August 16, 2021
"""

# imports standard libraries
import argparse
import os
import matplotlib.pyplot as plt 
from matplotlib import cm
import pickle as pkl
import numpy as np

# imports 3rd party libraries 
import cv2
import skimage.filters

# adds path to custom libraries
import sys 
sys.path.append('../src/')
# imports custom libraries
import genl.readin as readin
import genl.main_helper as mh
import cvimproc.basic as basic 

import config as cfg


def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Check quality of highlighting objects.')
    ap.add_argument('-b', '--brightness', default=2.0, type=float,
                    help='Factor to multiply image brightness by.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no apostrophes).')
    ap.add_argument('-i', '--input_file', default='input.txt',
                    help='Name of file with input parameters.')
    ap.add_argument('-q', '--quiet', default=False, type=int,
                    help='If True, will not print out success report upon saving image.')
    ap.add_argument('-p', '--prefix', default='intensity',
                    help='Prefix to saved images (<prefix>_<#>.<ext>)')
    ap.add_argument('-r', '--r_selem', default=2, type=int,
                    help='Radius of structuring element (e.g. for median filter before gradient calculation).')
    args = vars(ap.parse_args())

    return args


def compute_centroid(I, peak_frac=0.9):
    """
    Computes the centroid of an intensity profile by identifying
    the pixels near the peak and computing their centroid.
    """
    I_max = np.max(I)
    peak_pixels = []
    for row in range(I.shape[0]):
        for col in range(I.shape[1]):
            if I[row, col] > peak_frac*I_max:
                peak_pixels += [(row, col)]
    # computes centroid by taking mean of rows and columns
    peak_pixels_arr = np.asarray(peak_pixels)
    rc, cc = np.mean(peak_pixels_arr, axis=0)

    return rc, cc

def compute_gradients(I, centroid):
    """
    Computes gradients along four cardinal directions from centroid to edges.
    """
    rc, cc = [int(coord) for coord in centroid]

    ### Vertical gradient
    dIdrow = np.diff(I, axis=0)
    # gradient upwards from centroid (reversed direction requires negative sign)
    grad_up = -dIdrow[:rc]
    # gradient downwards from centroid
    grad_down = dIdrow[rc:]

    ### Horizontal gradient
    dIdcol = np.diff(I, axis=1)
    # gradient to the left of the centroid (reversed direction requires negative sign)
    grad_left = -dIdcol[:cc]
    # gradient to the right of the centroid
    grad_right = dIdcol[cc:]

    return grad_up, grad_down, grad_right, grad_left


def max_abs(val_arr_list):
    """Computes the maximum of the absolute values of the given list of arrays."""
    return np.max([np.max(np.abs(val_arr)) for val_arr in val_arr_list])


def main():
    # parses user-supplied information to identify data file for desired experiment
    args = parse_args()
    brightness = args['brightness']
    ext = args['ext']
    input_file = args['input_file']
    quiet = args['quiet']
    prefix = args['prefix']
    r_selem = args['r_selem']

    # loads parameters from input file
    p = readin.load_params(os.path.join(cfg.input_dir, input_file))

    # chooses end frame to be last frame if given as -1
    vid_path =  os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
    p['end'] = basic.get_frame_count(vid_path, p['end'])
    # creates filepaths for video, data, and figures
    _, data_path, vid_dir, \
    data_dir, figs_dir, _ = mh.get_paths(p, True)
   
    # tries to open data file (pkl)
    try:
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
            metadata = data['metadata']
            objs = data['objects']
            frame_IDs = data['frame IDs']
    except:
        print('Failed to load {0:s}--is there a file at this location?'.format(data_path))
        return

    # extracts background
    bkgd = metadata['bkgd'].astype(int)

    # loads video
    cap = cv2.VideoCapture(vid_path)

    # loops through objects
    for ID, obj in objs.items():

        # gets list of frame numbers in which the object appears
        frame_nums = obj.get_props('frame')

        # loops through frames 
        for n in frame_nums:

            # creates plot (two subplots side-by-side)
            fig = plt.figure()


            ### DISPLAYS IMAGE OF OBJECT
            ax_im = fig.add_subplot(2, 2, 1)

            # extracts frame from video
            frame = basic.read_frame(cap, n)
            # cuts off rows outside mask
            row_lo = metadata['row_lo']
            row_hi = metadata['row_hi']
            frame = frame[row_lo:row_hi, :]

            # extracts bounding box of object from frame
            row_min, col_min, row_max, col_max = obj.get_prop('bbox', n)

            # plots bbox of frame on left side of plot
            ax_im.imshow(basic.adjust_brightness(frame[row_min:row_max, col_min:col_max, :], brightness))
            ax_im.axis('off')


            ### DISPLAYS 3D SURFACE PLOT OF INTENSITY OF BKGD-SUB OBJECT
            # subtracts background
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(int)
            bkgd_sub = frame - bkgd 

            # creates mesh grid of bbox
            X = np.arange(col_min, col_max)
            Y = np.arange(row_min, row_max)
            X, Y = np.meshgrid(X, Y)
            I = -bkgd_sub[row_min:row_max, col_min:col_max] # flips background-subtracted intensity for easier seeing

            # computes centroid
            rc, cc = compute_centroid(I)
            # plots centroid on image
            ax_im.plot(cc, rc, '*', ms=10, color='r')

            # computes gradients (applies median filter to reduce noise)
            selem = skimage.morphology.selem.disk(r_selem)
            grads = compute_gradients(skimage.filters.median(I, selem), (rc, cc))
            # computes maximum gradient
            grad_max = max_abs(grads)

            # Also, check out unsharp masking with skimage here: ***HERE***
            # https://scikit-image.org/docs/dev/auto_examples/filters/plot_unsharp_mask.html

            # plots 3D surface plot on right side of plot
            ax_prof = fig.add_subplot(2, 2, 2, projection='3d')
            ax_prof.plot_surface(X, Y, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax_prof.set_title('Max grad = {0:d}'.format(int(grad_max)))


            ### DISPLAYS EFFECT OF SHARPNESS ON SHARPNESS METRIC
            radius_list = [1, 3, 5, 10]
            amount_list = [0.1, 0.5, 1, 2]
            ax_sharp = fig.add_subplot(2, 2, 3)
            
            for radius in radius_list:
                grad_max_list = []
                for amount in amount_list:
                    I_sharpened = skimage.filters.unsharp_mask(
                                                            I, 
                                                            radius=radius, 
                                                            amount=amount
                                                            )
                    # evaluate maximum gradient
                    grads = compute_gradients(
                                skimage.filters.median(I_sharpened, selem), 
                                (rc, cc)
                                )
                    # computes maximum gradient
                    grad_max_list += [max_abs(grads)]

                # plots sharpness metric vs. sharpness parameter
                ax_sharp.plot(amount_list, grad_max_list, 
                            label='radius = {0:d}'.format(int(radius)))


            ### INTEGRATES INTENSITY TO DIFFERENT RADII FROM CENTROID
            I_integ_list = []
            #for R in R_list:
                #integrate bkgd-sub intensity I to radius R
            #     I_integ_list += [I_integ]

            # ax_integ = fig.add_subplot(2, 2, 4)
            # ax_integ.plot(R_list, I_integ_list, 'b--')


            # saves plot
            plt.savefig(os.path.join(figs_dir, '{0:s}_obj{1:d}_f{2:d}.{3:s}'.format(prefix, ID, n, ext)))

            # prints out success
            if not quiet:
                print('Saved object in frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(n, 
                                                p['start'], p['every'], p['end']))

            plt.close()

    return 


if __name__ == '__main__':
    main()



