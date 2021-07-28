"""
test_superpose_images.py tests a method used to superpose images of an object
on the original background.

Based on `superpose_images()` method written in `test_classification.py`.
N.B.: uses `highlight_image()` method from `analysis/highlight.py`.

Author: Andy Ylitalo
Date: July 22, 2021
"""

# standard libraries
import os 
import pickle as pkl
import cv2
from matplotlib import cm
import numpy as np

# 3rd party libraries
import argparse

# custom libraries
import sys 
sys.path.append('../src/')
import cvimproc.improc as improc 
import cvimproc.basic as basic 
import cvimproc.mask as mask
import genl.readin as readin 
import genl.main_helper as mh
import config as cfg
# highlight methods
sys.path.append('../analysis/')
import highlight



###################### ARGUMENT PARSER ##############################

def parse_args():
    """Parses arguments provided in command line into function parameters."""
    ap = argparse.ArgumentParser(
        description='Superposes images of objects onto background frame.')
    ap.add_argument('-o', '--skip_overlaps', default=0, type=int,
                    help='If 1, skips superposing images that overlap with the previous.')
    ap.add_argument('-e', '--ext', default='jpg',
                    help='Extension for saved images (no periods).')
    ap.add_argument('-f', '--false_color', default=0, type=int,
                    help='If 1, false-colors image. color_objs should be 0.')
    # ap.add_argument('-i', '--input_file', default='input.txt',
    #                 help='Name of file with input parameters.')
    ap.add_argument('-s', '--save', default=1, type=int,
                    help='If 1, saves image.')
    ap.add_argument('-c', '--color_objs', default=0, type=int,
                    help='If 1, colors objects using highlight method.')
    ap.add_argument('-n', '--every', default=1, type=int, help='Superposes every nth image.')
    ap.add_argument('-r', '--disp_R', default=0, type=int,
                    help='If 1, will display radius [um] above object.')
    args = vars(ap.parse_args())

    skip_overlaps = bool(args['skip_overlaps'])
    ext = args['ext']
    false_color = bool(args['false_color'])
    save = bool(args['save'])
    color_objs = bool(args['color_objs'])
    every = int(args['every'])
    disp_R = bool(args['disp_R'])

    return skip_overlaps, ext, false_color, save, color_objs, every, disp_R


###################### HELPER FUNCTIONS #############################

def get_figs_dir(input_filepath):
    """Returns directory where figures are saved."""
    # loads parameters
    p = readin.load_params(input_filepath)
    # determines appropriate directories based on parameters
    _, _, _, _, figs_dir, _ = mh.get_paths(p)

    return figs_dir


def load_objs(input_filepath):
    """
    Loads objects pointed to by input filepath.

    Parameters
    ----------
    input_filepath : string
        Path to input file (stored in `data` subfolder)

    Returns
    -------
    objs : dictionary
        Dictionary of TrackedObj objects
    """
    # loads parameters
    p = readin.load_params(input_filepath)
    # determines appropriate directories based on parameters
    _, data_path, _, _, _, _ = mh.get_paths(p)

    # loads data
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
        metadata = data['metadata']
        objs = data['objects']

    return objs, metadata


def superpose_images(obj, metadata, skip_overlaps=False, 
                    num_frames_for_bkgd=100, every=1,
                    color_objs=False, disp_R=False, b=1.7,
                    false_color=False, cmap='jet'):
    """
    Superposes images of an object onto one frame.

    Parameters
    ----------
    vid_path : string
        Path to video in which object was tracked. Source folder is `src/`
    obj : TrackedObject
        Object that has been tracked. Must have 'image', 'local centroid',
        'frame_dim', 'bbox', and 'centroid' parameters.
    skip_overlaps : bool, optional
        If True, will skip superposing images that overlap with each other
        to produce a cleaner, though incomplete, image. Default False.
    every : int, optional
        Superposes every `every` image (so if every = 1, superposes every image;
        if every = 2, superposes every *other* image; if every = n, superposes every
        nth image). Default = 1
    color_objs : bool, optional
        If True, will use image processing to highlight objects in each frame
        before superposing. Default False
    disp_R : bool, optional
        If True, will display the radius measured by image-processing in um
        above each object.
    b : float, optional
        Factor by which to scale brightness of superposed images to match background.
        Not sure why they don't automatically appear with the same brightness.
        Default is 1.7.

    Returns
    -------
    im : (M x N) numpy array of uint8
        Image of object over time; each snapshot is superposed 
        (likely black-and-white)
    """
    ### initializes image as background ###
    # loads parameters
    highlight_kwargs = metadata['highlight_kwargs']
    mask_data = highlight_kwargs['mask_data']
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
    # computes background
    bkgd = improc.compute_bkgd_med_thread(metadata['vid_path'],
            vid_is_grayscale=True,  #assumes video is already grayscale
            num_frames=num_frames_for_bkgd,
            crop_y=row_lo,
            crop_height=row_hi-row_lo)

    # copies background to superpose object images on
    im = np.copy(bkgd)

    # converts image to 3-channel if highlighting objects (needs color)
    if color_objs:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    # initializes previous bounding box
    bbox_prev = (0,0,0,0)
    
    # loads video capture object
    cap = cv2.VideoCapture(metadata['vid_path'])

    # gets list of frames with object
    frame_list = obj.get_props('frame')

    ### Superposes image from each frame ###
    ct = 0
    for i, f in enumerate(frame_list):
        # only superposes every "every"th image
        if (ct % every) != 0:
            ct += 1
            continue

        # loads bounding box and image within it
        bbox = obj.get_prop('bbox', f)

        # skips images that overlap if requested
        if skip_overlaps:
            if basic.is_overlapping(bbox_prev, bbox):
                continue
            else:
                bbox_prev = bbox

        # highlights objects if requested
        if color_objs:
            # extracts radius of object
            R = obj.get_prop('radius [um]', f)

            # not sure why, but brightness must be 1.5 to match rest of image
            # selects offset that pushes label out of image 
            offset = bbox[3]-bbox[1]+5

            im_obj = highlight.highlight_image(basic.read_frame(cap, f), 
                                                f, cfg.highlight_method,
                                                metadata, {R : obj}, [R],
                                                brightness=b, offset=offset)

            # shows number ID of object in image
            centroid = obj.get_prop('centroid', f)
            # converts centroid from (row, col) to (x, y) for open-cv
            x = int(centroid[1])
            y = int(centroid[0])
            

            # superposes object image on overall image (3-channel images)
            row_min, col_min, row_max, col_max = bbox
            im[row_min:row_max, col_min:col_max, :] = im_obj[row_min:row_max, 
                                                            col_min:col_max, :]

            if disp_R:
                # prints label on image (radius [um])
                im = cv2.putText(img=im, text='{0:.1f}'.format(R), org=(x-10, y-7),
                                        fontFace=0, fontScale=0.5, color=cfg.white,
                                        thickness=2)
        else:
            # loads image
            im_raw = basic.read_frame(cap, f)
            im_obj = cv2.cvtColor(basic.adjust_brightness(im_raw, b), cv2.COLOR_BGR2GRAY)[row_lo:row_hi, :]

            # superposes object image on overall image
            row_min, col_min, row_max, col_max = bbox
            im[row_min:row_max, col_min:col_max] = im_obj[row_min:row_max, 
                                                            col_min:col_max]

        # increments counter 
        ct += 1

    # false-colors objects by taking signed difference with background
    if false_color:
        signed_diff = im.astype(int) - bkgd.astype(int)
        # remove noise above 0 (*assumes object is darker than background)
        if remove_positive_noise:
            signed_diff[signed_diff > 0] = 0
        # defines false-color mapping to range to max difference
        max_diff = np.max(np.abs(signed_diff))
        # normalizes image so -max_diff -> 0 and +max_diff -> 1
        im_norm = (signed_diff + max_diff) / (2*max_diff)
        # maps normalized image to color image (still as floats from 0 to 1)
        color_mapped = cm.get_cmap(cmap)(im_norm)
        # converts to OpenCV format (uint8 0 to 255)
        im_false_color = basic.cvify(color_mapped)
        # converts from RGBA to RGB
        im = cv2.cvtColor(im_false_color, cv2.COLOR_RGBA2RGB)

    return im



def main():
    # parses input arguments
    skip_overlaps, ext, false_color, save, color_objs, every, disp_R = parse_args()

    # inputs TODO make these parsed by argparse when running command
    input_subpaths = ['ppg_co2/20210720_70bar/ppg_co2_60000_000-7_050_0230_67_10_32/size1/data/input.txt']

#['ppg_co2/20210720_70bar/ppg_co2_60000_000-7_050_0230_67_10_32/size1/data/input.txt']

    # gets full filepath for each input file
    input_filepaths = [os.path.join(cfg.output_dir, input_subpath) for input_subpath in input_subpaths]

    # creates superposed image of objects for each input file
    for input_filepath in input_filepaths:
        print('Analyzing data from {0:s}'.format(input_filepath))

        # loads objects for current video
        objs, metadata = load_objs(input_filepath)

        # creates a new image of superposed frames for each object
        for ID, obj in objs.items():
            im = superpose_images(obj, metadata, skip_overlaps=skip_overlaps, every=every, 
                                    color_objs=color_objs, disp_R=disp_R, false_color=false_color)
            if save:
                # saves image
                suffix = ''
                if false_color:
                    suffix += '_false'
                if color_objs:
                    suffix += 'color'
                if disp_R:
                    suffix += '_R'
                if every != 1:
                    suffix += '_{0:d}'.format(every)
                save_path = os.path.join(get_figs_dir(input_filepath), 
                                            'obj{0:d}{1:s}.{2:s}'.format(ID, suffix, ext))
                
                basic.save_image(im, save_path)    

    return

if __name__ == '__main__':
    main()
