"""
highlight.py
@brief tests the success of highlighting objects by saving images of objects
highlighted using the current method. Great for quality control and trouble-
shooting good image-processing parameters.

@author Andy Ylitalo
@date October 24, 2020
"""

# standard libraries
import os
import pickle as pkl
import cv2
import numpy as np
import argparse

# 3rd party image-processing libraries
import skimage.measure
import skimage.color
import PIL.Image

# custom libraries
import sys
sys.path.append('../src/') # adds custom library directory to path
import cvimproc.basic as basic
import cvimproc.improc as improc
import genl.fn as fn
import genl.readin as readin

# imports configuration file
import config as cfg

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Check quality of highlighting objects.')
    ap.add_argument('-s', '--skip_blanks', default=1,
                    help='If 1, skips images without objects detected.')
    ap.add_argument('-b', '--brightness', default=3.0, type=float,
                    help='Factor to multiply image brightness by.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no apostrophes).')
    ap.add_argument('-i', '--input_file', default='input.txt',
                    help='Name of file with input parameters.')
    ap.add_argument('-c', '--color_object', default=1,
                    help='If 1, objects will be colored in figure.')
    ap.add_argument('--offset', default=5, type=int, help='Offset of labels to the right.')
    args = vars(ap.parse_args())

    return args

# TODO add option to save image with dimensions on tick marks

def main():
    # parses user-supplied information to identify data file for desired experiment
    args = parse_args()
    skip_blanks = args['skip_blanks']
    brightness = args['brightness']
    ext = args['ext']
    input_file = args['input_file']
    color_object = args['color_object']
    offset = args['offset']

    # loads data file and parameters from mask and input.txt files
    p = readin.load_params(os.path.join(cfg.input_dir, input_file))
    # defines filepath to video
    vid_path = os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
    # defines directory to video data and figures
    vid_dir = os.path.join(p['vid_subdir'], p['vid_name'][:-4], p['input_name'])
    data_dir = os.path.join(cfg.output_dir, vid_dir, cfg.data_subdir)
    figs_dir = os.path.join(cfg.output_dir, vid_dir, cfg.figs_subdir)

    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'], 
                                                    p['every'], p['end']))
    # tries to open data file (pkl)
    try:
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
            metadata = data['metadata']
            objects = data['objects']
            frame_IDs = data['frame IDs']
    except:
        print('Failed to load {0:s}--is there a file at this location?'.format(data_path))
        return

    # loads video
    cap = cv2.VideoCapture(vid_path)
    # chooses end frame to be last frame if given as -1
    if p['end'] == -1:
        p['end'] = basic.count_frames(vid_path)

    # loops through frames of video with objects according to data file
    for f in range(p['start'], p['end'], p['every']):

        # skips frame upon request if no objects identified in frame
        if len(frame_IDs[f]) == 0 and skip_blanks:
            continue
        # loads frame
        frame = basic.read_frame(cap, f)
        # crops frame
        row_lo = metadata['row_lo']
        row_hi = metadata['row_hi']
        frame = frame[row_lo:row_hi, :]
        # extracts value channel
        val = basic.get_val_channel(frame)

        # highlights object according to parameters from data file
        bkgd = metadata['bkgd']
        obj = p['highlight_method'](val, bkgd, **metadata['args'])
        # applies highlights
        frame_labeled, num_labels = skimage.measure.label(obj, return_num=True)
        # OpenCV version--less convenient
        #num_labels, frame_labeled, _, _ = cv2.connectedComponentsWithStats(obj)

        # labels objects
        IDs = frame_IDs[f]
        frame_relabeled = np.zeros(frame_labeled.shape)
        for ID in IDs:
            # finds label associated with the object with this id
            rc, cc = objects[ID].get_prop('centroid', f)
            label = improc.find_label(frame_labeled, rc, cc)
            # re-indexes from 1-255 for proper coloration by label2rgb
            # (so 0 can be bkgd)
            new_ID = (ID % 255) + 1
            frame_relabeled[frame_labeled==label] = new_ID

        # brightens original image
        frame_adj = basic.adjust_brightness(frame, brightness)
        # colors in objects according to label (not consistent frame-to-frame)
        if color_object:
            frame_disp = fn.one_2_uint8(skimage.color.label2rgb(frame_relabeled,
                                                image=frame_adj, bg_label=0))
        else:
            frame_disp = fn.one_2_uint8(frame_relabeled)

        # prints ID number of object to upper-right of centroid
        # this must be done after image is colored
        for ID in IDs:
            # shows number ID of object in image
            centroid = objects[ID].get_prop('centroid', f)
            # converts centroid from (row, col) to (x, y) for open-cv
            x = int(centroid[1])
            y = int(centroid[0])
            # text of number ID is black if on the border of the image, white o/w
            on_border = objects[ID].get_prop('on border', f)
            outer_stream = objects[ID].get_props('inner stream') == 0
            error = objects[ID].get_props('inner stream') == -1
            if on_border or outer_stream:
                color = cfg.black
            elif not error:
                color = cfg.white
            else:
                color = cfg.red
            frame_disp = cv2.putText(img=frame_disp, text=str(ID), org=(x+offset, y),
                                    fontFace=0, fontScale=0.5, color=color,
                                    thickness=2)

        # adds scale bar if desired--TODO

        # saves image
        im = PIL.Image.fromarray(frame_disp)
        im.save(os.path.join(figs_dir, '{0:d}.{1:s}'.format(f, ext)))

        print('Saved frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(f, 
                                             p['start'], p['every'], p['end']))

    return


if __name__ == '__main__':
    main()
