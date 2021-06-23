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


###################### HELPER FUNCTIONS #################################
def determine_label_color(obj, f, std_color, border_color, error_color):
    """
    Determines the color of the label to be printed on the image.

    Parameters
    ----------
    obj : TrackedObject
        Object to be labeled in image
    f : int
        Frame number in video
    std_color, border_color, error_color : 3-tuple of uint8s, opt (default white, black, red)
        Colors for labels of standard objects, objects on the border, and objects that
        raised some kind of red flag or error

    Returns
    -------
    color : 3-tuple of uint8s
        Color of label (RGB)
    """
    # text of number ID is black if on the border of the image, white o/w
    on_border = obj.get_prop('on border', f)
    outer_stream = obj.get_props('inner stream') == 0
    error = obj.get_props('inner stream') == -1
    if on_border or outer_stream:
        color = border_color
    elif not error:
        color = std_color
    else:
        color = error_color

    return color


def label_objects_in_frame(objects, IDs, f, frame_labeled):
    """
    Labels the objects in the given frame based on their ID number
    (hopefully this ensures that the object remains labeled with the 
    same color in different frames, but it doesn't seem that robust).

    Parameters
    ----------
    objects : dictionary
        Dictionary of TrackedObject objects, indexed by ID #
    IDs : list of ints
        List of ID #s of objects in the given frame
    f : int
        frame number in video
    frame_labeled : (M x N) or (M x N x 3) numpy array of uint8s
        Frame in which the pixels of each object have the same unique
        pixel value. This value is arbitrary, so this function tries
        to make it correspond to the ID # of the object.
    
    Returns
    -------
    frame_relabeled : (M x N) or (M x N x 3) numpy array of uint8s
        Frame of the same size as the provided frame with new labels
        that correspond to the ID #s of the objects
    """
    # creates frame to hold the new labels
    frame_relabeled = np.zeros(frame_labeled.shape)

    # relabels each object by ID #
    for ID in IDs:

        # finds label associated with the object with this id
        rc, cc = objects[ID].get_prop('centroid', f)
        label = improc.find_label(frame_labeled, rc, cc)

        # re-indexes from 1-255 for proper coloration by label2rgb
        # (so 0 can be bkgd)
        new_ID = (ID % 255) + 1
        frame_relabeled[frame_labeled==label] = new_ID

    return frame_relabeled


################################## PRIMARY FUNCTIONS ###################################

def highlight_and_save_image(image, f, metadata, objects, frame_IDs,
                             brightness, ext, color_object, offset,
                             std_color, border_color, error_color):
    """
    Highlights and labels objects within given image, then saves.
    
    Parameters
    ----------
    image : (M x N) or (M x N x 3) numpy array of uint8s
        image to highlight and save (can be gray scale or color)
    f : int
        frame number in video
    metadata : dictionary
        Metadata of the video (see `src/main.py` for contents)
    objects : dictionary
        Dictionary of tracked objects, indexed by ID #
        See TrackedObject class in `classes/classes.py` for more details
    frame_IDs : dictionary
        Dictionary indexed by frame number of the ID #s of objects in that frame
    brightness : float, opt (default=3.0)
        Factor by which to scale brightness of images
    ext : string, opt (default='jpg')
        Extension (excluding '.') for images that are saved.
        Use extensions supported by the Python Imaging Library (PIL),
        such as 'jpg', 'png', and 'tif'.
    color_object : bool, opt (default=True)
        If True, shades in objects over the original image in an 
        arbitrary color
    offset : int, opt (default=5)
        Number of pixels by which to offset label to the right of the 
        object's centroid
    std_color, border_color, error_color : 3-tuple of uint8s, opt (default white, black, red)
        Colors for labels of standard objects, objects on the border, and objects that
        raised some kind of red flag or error

    Returns nothing.
    """
    # crops frame
    row_lo = metadata['row_lo']
    row_hi = metadata['row_hi']
    frame = image[row_lo:row_hi, :]
    # extracts value channel
    val = basic.get_val_channel(frame)

    # highlights object according to parameters from data file
    bkgd = metadata['bkgd']
    obj = p['highlight_method'](val, bkgd, **metadata['args'])
    # applies highlights
    frame_labeled, num_labels = skimage.measure.label(obj, return_num=True)
    # OpenCV version--less convenient
    #num_labels, frame_labeled, _, _ = cv2.connectedComponentsWithStats(obj)

    # labels objects in frame
    frame_relabeled = label_objects_in_frame(objects, frame_IDs[f], 
                                                f, frame_labeled)
    
    # brightens original image
    frame_adj = basic.adjust_brightness(frame, brightness)
    # colors in objects according to label (not consistent frame-to-frame)
    if color_object:
        frame_disp = fn.one_2_uint8(skimage.color.label2rgb(frame_relabeled,
                                            image=frame_adj, bg_label=0))
    else:
        frame_disp = fn.one_2_uint8(frame_relabeled)

    # prints ID number of object to the upper-right of the centroid
    # this must be done after image is colored
    for ID in IDs:

        # determines color of the label
        color = determine_label_color(objects[ID], f, std_color, 
                                border_color, error_color)

        # shows number ID of object in image
        centroid = obj.get_prop('centroid', f)
        # converts centroid from (row, col) to (x, y) for open-cv
        x = int(centroid[1])
        y = int(centroid[0])
        
        # prints label on image
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


def highlight_and_save_tracked_video(input_file, input_dir, output_dir, 
                            data_subdir, figs_subdir, skip_blanks=True,
                            brightness=3.0, ext='jpg', color_object=True,
                            offset=5, std_color=(255,255,255), 
                            border_color=(0,0,0), error_color=(255,0,0)):
    """
    Highlights objects in the video specified in the input file using data
    from the object-tracking in `src/main.py`. Then saves images.

    Parameters
    ----------
    input_file : string
        Name of file with input parameters
    input_dir : string
        Path to directory with input file
    output_dir : string
        Path to directory to save output to
    data_subdir : string
        Name of subdirectory for holding the data for this analysis.
        Data from that analysis will be loaded for object labeling.
    figs_subdir : string
        Name of subdirectory for saving the figures produced.
    skip_blanks : bool, opt (default=True)
        If True, does not save images that have no objects
    brightness : float, opt (default=3.0)
        Factor by which to scale brightness of images
    ext : string, opt (default='jpg')
        Extension (excluding '.') for images that are saved.
        Use extensions supported by the Python Imaging Library (PIL),
        such as 'jpg', 'png', and 'tif'.
    color_object : bool, opt (default=True)
        If True, shades in objects over the original image in an 
        arbitrary color
    offset : int, opt (default=5)
        Number of pixels by which to offset label to the right of the 
        object's centroid
    std_color, border_color, error_color : 3-tuple of uint8s, opt (default white, black, red)
        Colors for labels of standard objects, objects on the border, and objects that
        raised some kind of red flag or error

    Returns nothing.
    """
    # loads data file and parameters from mask and input.txt files
    p = readin.load_params(os.path.join(input_dir, input_file))
    # defines filepath to video
    vid_path = os.path.join(input_dir, p['vid_subdir'], p['vid_name'])
    # defines directory to video data and figures
    vid_dir = os.path.join(p['vid_subdir'], p['vid_name'][:-4], p['input_name'])
    data_dir = os.path.join(output_dir, vid_dir, data_subdir)
    figs_dir = os.path.join(output_dir, vid_dir, figs_subdir)

    # defines name of data file to save
    data_path = os.path.join(data_dir, 'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'], 
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
        # highlights and saves frame
        highlight_and_save_image(frame, f, metadata, objects, frame_IDs,
                             brightness, ext, color_object, offset,
                             std_color, border_color, error_color)

    return 

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

    highlight_and_save_tracked_video(input_file, cfg.input_dir, cfg.output_dir, 
                            cfg.data_subdir, cfg.figs_subdir, 
                            skip_blanks=skipe_blanks, brightness=brightness, 
                            ext=ext, color_object=color_object, offset=offset, 
                            std_color=cfg.white, border_color=cfg.black, error_color=cfg.red)

    return


if __name__ == '__main__':
    main()
