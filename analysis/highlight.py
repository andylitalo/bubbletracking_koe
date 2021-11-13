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

# custom libraries
import sys
sys.path.append('../src/') # adds custom library directory to path
import cvimproc.basic as basic
import cvimproc.improc as improc
import genl.fn as fn
import genl.readin as readin
import genl.main_helper as mh

# imports configuration file
import config as cfg

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Check quality of highlighting objects.')
    ap.add_argument('-s', '--skip_blanks', default=1, type=int,
                    help='If 1, skips images without objects detected.')
    ap.add_argument('-b', '--brightness', default=3.0, type=float,
                    help='Factor to multiply image brightness by.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no apostrophes).')
    ap.add_argument('-i', '--input_file', default='input.txt',
                    help='Name of file with input parameters.')
    ap.add_argument('-c', '--color_object', default=1, type=int,
                    help='If 1, objects will be colored in figure.')
    ap.add_argument('--offset', default=5, type=int, help='Offset of labels to the right.')
    ap.add_argument('-q', '--quiet', default=False, type=int,
                    help='If True, will not print out success report upon saving image.')
    ap.add_argument('-k', '--save_bkgd_sub', default=0, type=int,
                    help='If 1, saves background-subtracted image, too.')
    args = vars(ap.parse_args())

    return args


###################### HELPER FUNCTIONS #################################
def bubble_label_color(bubble, f, std_color=cfg.white, 
                        border_color=cfg.black, error_color=cfg.red):
    """
    Determines the color of the label to be printed on the image 
    for a Bubble object.

    Parameters
    ----------
    bubble : Bubble
        Bubble object to be labeled in image (has "inner_stream" property)
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
    # checks for errors
    is_bubble = bubble.get_prop('inner stream', f) and \
                    bubble.get_prop('solid', f) and \
                    bubble.get_prop('circular', f) and \
                    bubble.get_prop('oriented', f) and \
                    bubble.get_prop('consecutive', f) and \
                    bubble.get_prop('exited', f) and \
                    bubble.get_prop('growing', f)

    if not is_bubble:
        color = error_color 
    else:
        # text of number ID is black if on the border of the image, white o/w
        on_border = bubble.get_prop('on border', f)
        outer_stream = not bubble.get_prop('inner stream', f)
        if on_border or outer_stream:
            color = border_color
        else:
            color = std_color

    return color


def label_objects_in_frame(objs, IDs, f, frame_labeled):
    """
    Labels the objects in the given frame based on their ID number
    (hopefully this ensures that the object remains labeled with the 
    same color in different frames, but it doesn't seem that robust).

    Parameters
    ----------
    objs : dictionary
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
        rc, cc = objs[ID].get_prop('centroid', f)
        label = improc.find_label(frame_labeled, rc, cc)

        # re-indexes from 1-255 for proper coloration by label2rgb
        # (so 0 can be bkgd)
        new_ID = (ID % 255) + 1
        frame_relabeled[frame_labeled==label] = new_ID

    return frame_relabeled


################################## PRIMARY FUNCTIONS ###################################

def highlight_image(image, f, highlight_method, metadata, objs, IDs,
                    brightness=3.0, color_object=True, offset=5,
                    label_color_method=bubble_label_color, label_kwargs={}):
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
    objs : dictionary
        Dictionary of tracked objects, indexed by ID #
        See TrackedObject class in `classes/classes.py` for more details
    IDs : list
        List of the ID #s of objects in the present frame
    save_dir : string
        Directory in which to save the image
    brightness : float, opt (default=3.0)
        Factor by which to scale brightness of images
    color_object : bool, opt (default=True)
        If True, shades in objects over the original image in an 
        arbitrary color
    offset : int, opt (default=5)
        Number of pixels by which to offset label to the right of the 
        object's centroid
    label_color_method : functor, optional
        Method for determining color of label (default is `bubble_label_color`)
    label_kwargs : dictionary, optional
        Keyword arguments for label_color_method (besides object and frame number)
        
    Returns nothing.
    """
    # crops frame
    row_lo = metadata['row_lo']
    row_hi = metadata['row_hi']
    frame = image[row_lo:row_hi, :]
    # extracts value channel
    # val = basic.get_val_channel(frame)
    val = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # highlights object according to parameters from data file
    bkgd = metadata['bkgd']
    highlighted = highlight_method(val, bkgd, **metadata['highlight_kwargs'])
    # applies highlights
    frame_labeled, num_labels = skimage.measure.label(highlighted, return_num=True)
    # OpenCV version--less convenient
    #num_labels, frame_labeled, _, _ = cv2.connectedComponentsWithStats(obj)

    # labels objects in frame
    frame_relabeled = label_objects_in_frame(objs, IDs, f, frame_labeled)
    
    # brightens original image
    frame_adj = basic.adjust_brightness(frame, brightness)
    # colors in objects according to label (not consistent frame-to-frame)
    if color_object:
        frame_disp = fn.one_2_uint8(skimage.color.label2rgb(frame_relabeled,
                                            image=frame_adj, bg_label=0))
    else:
        frame_disp = frame_adj

    # prints ID number of object to the upper-right of the centroid
    # this must be done after image is colored
    for ID in IDs:
        # grabs object with present ID
        obj = objs[ID]

        # determines color of the label
        color = label_color_method(obj, f, **label_kwargs)

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

    return frame_disp


def highlight_and_save_tracked_video(p, input_dir, output_dir, 
                            data_subdir, figs_subdir, skip_blanks=True,
                            brightness=3.0, ext='jpg', color_object=True,
                            offset=5, quiet=False, label_color_method=bubble_label_color,
                            label_kwargs={}, save_bkgd_sub=False):
    """
    Highlights objects in the video specified in the input file using data
    from the object-tracking in `src/main.py`. Then saves images.

    Parameters
    ----------
    p : dictionary
        Parameters of analysis (typically loaded from `input.txt` file and read
        by `readin.py`)
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
    save_bkgd_sub : bool, optional
        If True, saves background-subtracted image. Default is False.

    Returns nothing.
    """
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

    # loads video
    cap = cv2.VideoCapture(vid_path)

    # loops through frames of video with objects according to data file
    for f in range(p['start'], p['end'], p['every']):

        # extracts IDs of objects in the present frame
        IDs = frame_IDs[f]
        # skips frame upon request if no objects identified in frame
        if len(IDs) == 0 and skip_blanks:
            continue

        # loads frame
        frame = basic.read_frame(cap, f)

        # highlights and labels objects in frame frame
        im_labeled = highlight_image(frame, f, cfg.highlight_method,
                                metadata, objs, IDs,
                                brightness, color_object, offset,
                                label_color_method=label_color_method,
                                label_kwargs=label_kwargs)
        # saves image
        save_path = os.path.join(figs_dir, '{0:d}.{1:s}'.format(f, ext))
        basic.save_image(im_labeled, save_path)

        # saves background-subtracted image if requested
        if save_bkgd_sub:
            # crops frame
            row_lo = metadata['row_lo']
            row_hi = metadata['row_hi']
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # highlights object according to parameters from data file
            bkgd = metadata['bkgd']
            bkgd_sub = cv2.absdiff(frame[row_lo:row_hi, :], bkgd)
            save_path = os.path.join(figs_dir, 'bkg_sub_{0:d}.{1:s}'.format(f, ext))
            basic.save_image(bkgd_sub, save_path)

        # prints out success
        if not quiet:
            print('Saved frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(f, 
                                            p['start'], p['every'], p['end']))

    return 

# TODO add option to save image with dimensions on tick marks

def main():
    # parses user-supplied information to identify data file for desired experiment
    args = parse_args()
    skip_blanks = args['skip_blanks']
    brightness = args['brightness']
    ext = args['ext']
    input_file = args['input_file']
    color_object = bool(args['color_object'])
    offset = args['offset']
    quiet = args['quiet']
    save_bkgd_sub = args['save_bkgd_sub']

    # loads parameters from input file
    p = readin.load_params(os.path.join(cfg.input_dir, input_file))

    # highlights and saves images in tracked video
    highlight_and_save_tracked_video(p, cfg.input_dir, cfg.output_dir, 
                            cfg.data_subdir, cfg.figs_subdir, 
                            skip_blanks=skip_blanks, brightness=brightness, 
                            ext=ext, color_object=color_object, offset=offset, 
                            quiet=quiet, save_bkgd_sub=save_bkgd_sub)

    return


if __name__ == '__main__':
    main()
