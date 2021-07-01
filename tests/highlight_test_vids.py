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

# local libraries
import readin

# imports configuration file
import config as cfg

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

    # loads data file and parameters from mask and input.txt files
    input_path = os.path.join(cfg.input_dir, input_file)


    for i in tests:
        # refreshes parameters that changed
        p = readin.load_params(input_path)
        # gets paths to data, reporting failure if occurs
        p['vid_name'] = str(i) + p['vid_ext']
        # if last frame given as -1, returns as final frame of video
        # *Note: must move up one directory to `src` for correct filepath
        vid_path =  os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
        p['end'] = basic.get_frame_count(vid_path, p['end'])

        _, data_path, vid_dir, \
        data_dir, figs_dir, _ = mh.get_paths(p, True)
      
        # defines name of data file to save
        p['end'] = basic.get_frame_count(vid_path, p['end'])

        # tries to open data file (pkl)
        try:
            with open(data_path, 'rb') as f:
                data = pkl.load(f)
                metadata = data['metadata']
                bubbles = data['objects']
                frame_IDs = data['frame IDs']
        except:
            print('No file at specified data path {0:s}.'.format(data_path))
            return

        # loads video
        cap = cv2.VideoCapture(vid_path)

        # loops through frames of video with bubbles according to data file
        for f in range(p['start'], p['end'], p['every']):

            # skips frame upon request if no bubbles identified in frame
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

            # highlights bubble according to parameters from data file
            bkgd = metadata['bkgd']
            highlighted = cfg.highlight_method(val, bkgd, **metadata['highlight_kwargs'])

            # applies highlights
            frame_labeled, num_labels = skimage.measure.label(highlighted, return_num=True)
            #num_labels, frame_labeled, _, _ = cv2.connectedComponentsWithStats(bubble)

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
            if color_object:
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
                # won't look for bubbles in outer stream, so all text will be white
                color = cfg.white
                frame_disp = cv2.putText(img=frame_disp, text=str(ID), org=(x, y),
                                        fontFace=0, fontScale=1, color=color,
                                        thickness=2)

            # adds scale bar if desired--TODO
            
            # saves image
            im = PIL.Image.fromarray(frame_disp)
            im.save(os.path.join(figs_dir, '{0:d}.{1:s}'.format(f, ext)))

            print('Saved frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(f, p['start'], p['every'], p['end']))

    return


if __name__ == '__main__':
    main()