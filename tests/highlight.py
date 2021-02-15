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
sys.path.append('../../libs/')
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
import genl.fn as fn
# imports from trackbubble
import genl.readin as readin

# GLOBAL VARIABLES
white = (255, 255, 255)
black = (0, 0, 0)
input_folder = '../input/' # relative path to input file


def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Check quality of highlighting bubbles.')
    ap.add_argument('-s', '--skip_blanks', default=0,
                    help='If 1, skips images without bubbles detected.')
    ap.add_argument('-b', '--brightness', default=3.0, type=float,
                    help='Factor to multiply image brightness by.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no apostrophes).')
    ap.add_argument('-i', '--input_file', default='input.txt',
                    help='Name of file with input parameters.')
    ap.add_argument('-c', '--color_bubble', default=1,
                    help='If 1, bubbles will be colored in figure.')
    args = vars(ap.parse_args())

    return args

# TODO add option to save image as bokeh plot with dimensions on tick marks

def main():
    # parses user-supplied information to identify data file for desired experiment
    args = parse_args()
    skip_blanks = args['skip_blanks']
    brightness = args['brightness']
    ext = args['ext']
    input_file = args['input_file']
    color_bubble = args['color_bubble']

    # loads data file and parameters from mask and input.txt files
    input_path = input_folder + input_file
    params = readin.load_params(input_path)
    input_name, eta_i, eta_o, L, R_o, selem, width_border, \
    fig_size_red, num_frames_for_bkgd, \
    start, end, every, th, th_lo, th_hi, \
    min_size_hyst, min_size_th, min_size_reg, \
    highlight_method, \
    vid_subfolder, vid_name, \
    expmt_folder, data_folder, fig_folder = params
    # defines filepath to video
    vid_path = expmt_folder + vid_subfolder + vid_name
    # defines directory to video data and figures
    vid_dir = vid_subfolder + os.path.join(vid_name[:-4], input_name)
    data_dir = data_folder + vid_dir
    fig_dir = fig_folder + vid_dir

    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(start, every, end))
    # tries to open data file (pkl)
    try:
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
            metadata = data['metadata']
            bubbles = data['bubbles']
            frame_IDs = data['frame IDs']
    except:
        print('No file at specified data path.')
        return

    # loads video
    cap = cv2.VideoCapture(vid_path)
    # chooses end frame to be last frame if given as -1
    if end == -1:
        end = vid.count_frames(vid_path)

    # loops through frames of video with bubbles according to data file
    for f in range(start, end, every):

        # skips frame upon request if no bubbles identified in frame
        if len(frame_IDs[f]) == 0 and skip_blanks:
            continue
        # loads frame
        frame = basic.read_frame(cap, f)
        # extracts value channel
        val = basic.get_val_channel(frame)

        # highlights bubble according to parameters from data file
        bubble = highlight_method(val, metadata['bkgd'], *metadata['args'])
        # applies highlights
        frame_labeled, num_labels = skimage.measure.label(bubble, return_num=True)

        # labels bubbles
        IDs = frame_IDs[f]
        frame_relabeled = np.zeros(frame_labeled.shape)
        for ID in IDs:
            # finds label associated with the bubble with this id
            rc, cc = bubbles[ID].get_prop('centroid', f)
            label = improc.find_label(frame_labeled, rc, cc)
            # re-indexes from 1-255 for proper coloration by label2rgb
            # (so 0 can be bkgd)
            new_ID = (ID % 255) + 1
            frame_relabeled[frame_labeled==label] = new_ID

        # brightens original image
        frame_adj = basic.adjust_brightness(frame, brightness)
        # colors in bubbles according to label (not consistent frame-to-frame)
        if color_bubble:
            frame_disp = fn.one_2_uint8(skimage.color.label2rgb(frame_relabeled,
                                                image=frame_adj, bg_label=0))
        else:
            frame_disp = fn.one_2_uint8(frame_relabeled)

        # prints ID number of bubble to upper-right of centroid
        # this must be done after image is colored
        for ID in IDs:
            # shows number ID of bubble in image
            centroid = bubbles[ID].get_prop('centroid', f)
            # converts centroid from (row, col) to (x, y) for open-cv
            x = int(centroid[1])
            y = int(centroid[0])
            # text of number ID is black if on the border of the image, white o/w
            on_border = bubbles[ID].get_prop('on border', f)
            outer_stream = bubbles[ID].get_props('inner stream') == 0
            if on_border or outer_stream:
                color = black
            else:
                color = white
            frame_disp = cv2.putText(img=frame_disp, text=str(ID), org=(x, y),
                                    fontFace=0, fontScale=2, color=color,
                                    thickness=3)

        # adds scale bar if desired--TODO

        # saves image
        im = PIL.Image.fromarray(frame_disp)
        im.save(os.path.join(fig_dir, '{0:d}.{1:s}'.format(f, ext)))

        print('Saved frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(f, start, every, end))

    return


if __name__ == '__main__':
    main()
