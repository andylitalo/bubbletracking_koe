"""
demos.py contains methods used at Dow Technology Day 2021
to demonstrate different features of the image-processing
pipeline of bubbletracking_koe, based on the CvVidProc library.

Currently excludes methods for demo6.py (filtering artifacts from
objects and plotting histograms of the resulting objects).

*Note: requires Bokeh for plotting (tested with v1.4.0).

Author: Andy Ylitalo
Date: June 11, 2021
"""


# Python libraries
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import pickle as pkl
import glob
import shutil

# 3rd party libraries
import imageio
# bokeh
from bokeh.plotting import figure
from bokeh.io import show
# image-processing libraries
import skimage.morphology
import skimage.measure
import skimage.color
import scipy.ndimage

# Custom libraries
# first, specify paths to search for libraries
import sys
# second, import custom libraries
sys.path.append('../src/')
# source library
import cvimproc.improc as improc
import cvimproc.basic as basic
import cvimproc.pltim as pltim
import cvimproc.mask as mask
import genl.fn as fn
import genl.flow as flow
import config as cfg
# conversions
from genl.conversions import *

# analysis library
import statlib as stat
import demos_helper as dh



##################### DEMO 2 - HOW TO DETECT OBJECTS ###########################

def compare_bkgd_algos(vid_path, num_frames_for_bkgd, mag=4, 
                       scale_fig=1, brightness=1, show_fig=True):
    """
    Compares the results of three different algorithms for 
    computing the background of an image:
    1) selecting a frame (in this case, the first one)
    2) taking the mean of many frames
    3) taking the median of many frames (speed-up by `CvVidProc` library)
    
    Uses Bokeh library to show results in linked frames, such 
    that zooming in or panning on one image will do the same 
    for the others.
    
    Assumes Photron camera videos.
    
    Parameters
    ----------
    vid_path : string
        Path to the video whose background you want to calculate (.mp4 supported)
    num_frames_for_bkgd : int
        number of frames to use to compute background (for mean and median algos only)
    mag : int, opt (default=4)
        Magnification of the microscope--used to look up the conversion of pixels to microns
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of a figure
    brightness : float, opt (default=1)
        Factor by which to scale the brightness of a figure
    show_fig : bool, opt (default=False)
        If True, automatically shows figure
        
    Returns
    -------
    p_grid : bokeh plot object
        Grid of background images linked in zooming/panning; show with bokeh.io.show(p_grid)
    """
    # Derived Parameters
    mask_path = vid_path[:-4] + '_mask.pkl'
    pix_per_um = pix_per_um_photron[mag]

    # 1) Extracts object-free frame
    start_time = time.time()
    first_frame, ret = basic.load_frame(vid_path, 0)
    print('Extracted first frame in {0:.3f} s'.format(time.time() - start_time))
    
    # 2) Computes mean
    start_time = time.time()
    bkgd_mean = improc.compute_bkgd_mean(vid_path, num_frames=num_frames_for_bkgd, print_freq=10000)
    print('Computed mean in {0:.3f} s'.format(time.time() - start_time))

    # 3) Computes median (timing is printed automatically)
    print('Computing median...')
    bkgd_med = improc.compute_bkgd_med_thread(vid_path, vid_is_grayscale=True, num_frames=num_frames_for_bkgd)

    # Formats and displays the different methods for computing the background
    frame_list = [pltim.bokehfy(first_frame), pltim.bokehfy(bkgd_mean), pltim.bokehfy(bkgd_med)]
    title_list = ['First Frame', 'Mean', 'Median']
    p_grid = pltim.linked_frames(frame_list, pix_per_um, scale_fig, shape=(len(frame_list),1), 
                                 title_list=title_list, show_fig=show_fig, brightness=brightness)
    
    return p_grid


def false_color_bkgd_sub(vid_path, num, x_range, 
                         num_frames_for_bkgd=100, bkgd=None, 
                         scale_fig=1, scale_height=1.4, title='', mag=4, 
                         show_fig=True, t_fs=24, ax_fs=16, tk_fs=12, cb_fs=14):
    """
    Computes the background (median) and subtracts from the requested frame. 
    Adds false color since values of the background-subtracted frame can be 
    negative (while image pixel values must range from 0 to 255).
    Plots false-color image in Bokeh for interactivity.
    
    Assumes Photron camera videos.
    
    Parameters
    ----------
    vid_path : string
        Path to the video whose background you want to calculate (.mp4 supported)
    num : int
        Number of the frame to load for background subtraction
    x_range : 2-tuple of ints
        Range of x-values (xmin, xmax) to show in plot window (xmin >= 0, xmax < width in microns)
    num_frames_for_bkgd : int, opt (default=100)
        number of frames to use to compute background (for mean and median algos only)
    bkgd : (M x N) array of uint8, opt (default=None)
        If provided, this function will use it as the background instead of computing from scratch
    scale_fig : float, opt (default=1)
        Factor by which to scale the figure size for convenient viewing
    scale_height : float, opt (default=1.4)
        Separate scaling factor for the height to correct for Bokeh's tendency to squish images vertically
    title : string, opt (default='')
        Title of figure
    mag : int, opt (default=4)
        Magnification of the microscope--used to look up the conversion of pixels to microns
    show_fig : bool, opt (default=False)
        If True, automatically shows figure
    t_fs, ax_fs, tk_fs, cb_fs : int, opt (default=24, 16, 12, 14)
        Font sizes (title, axis labels, tick mark labels, color bar labels)
        
    Returns
    -------
    p : Bokeh figure
        Figure showing false-color background-subtracted image
    signed_diff : (M x N) array of int
        Background-subtracted image with the sign preserved
    bkgd : (M x N) array of uint8
        Background (computed by taking median)
    """
    ### BACKGROUND SUBTRACTION ###
    signed_diff = basic.bkgd_sub_signed(vid_path, num, bkgd=bkgd, 
                                         num_frames_for_bkgd=num_frames_for_bkgd)
    
    ### PLOT BKGD-SUBTRACTED IMAGE ###
 
    # looks up the conversion from pixels to microns
    pix_per_um = pix_per_um_photron[mag]
    # converts dimensions of image from pixels to length
    width_um = int(signed_diff.shape[1] / pix_per_um)
    height_um = int(signed_diff.shape[0] / pix_per_um)
    
    # creates Bokeh figure (based on format_frame())
    p = pltim.bokeh_fig(signed_diff, x_range, (0, height_um), 
                        title, scale_fig, scale_height, 
                        xlabel='x [um]', ylabel='y [um]')
    # defines false-color mapping to range to max difference
    max_diff = np.max(np.abs(signed_diff))
    # adds colorbar to figure
    p, color_mapper = dh.add_colorbar(p, "Turbo256", -max_diff, max_diff, cb_fs=cb_fs)
    # shows image on plot
    p.image(image=[signed_diff],             # image data
             x=0,               # lower left x coord
             y=0,               # lower left y coord
             dw=width_um,        # *data space* width of image
             dh=height_um,        # *data space* height of image
             color_mapper=color_mapper,    # palette name
    )
    # adjusts font sizes
    p = dh.set_bokeh_fig_fonts(p, t_fs, ax_fs, tk_fs)

    if show_fig:
        show(p)
        
    return p, signed_diff, bkgd


def compare_roc(vid_path, true_objects_path):
    """
    Compares the receiver operating characteristic (ROC) curve for
    different detection metrics: mean, standard deviation, and minimum 
    of background subtracted images. Plots ROC for each.
    
    Parameters
    ----------
    vid_path : string
        Filepath to the video to analyze
    true_objects_path : string
        Filepath to csv file containing list of frames with a true object
        
    Returns
    -------
    bkgd : (M x N) numpy array of uint8s
        Background computed from the video using the median    
    """
    # counts the number of frames in the video
    num_frames = basic.count_frames(vid_path)
    # computes background (median)
    bkgd = improc.compute_bkgd_med_thread(vid_path,
            vid_is_grayscale=True,
            num_frames=num_frames)
    # computes the mean, mean^2, standard deviation, and minimum value of each background-subtracted frame
    mean_list, mean_sq_list, stdev_list, min_val_list = stat.proc_stats(vid_path, bkgd, num_frames-1)

    # loads array of true objects (labeled by eye manually)
    true_objects = np.genfromtxt(true_objects_path, dtype=int, delimiter=',')
    # produces binary vector of labels
    y = stat.inds_to_labels(true_objects, num_frames)

    ### COMPUTES ROC FOR DIFFERENT METRICS ###
    # converts each metric so that higher values corespond to positive label
    x_list = [-np.asarray(mean_list), 
                np.asarray(stdev_list), 
                -np.asarray(min_val_list)]
    x_names = ['mean', 'stdev', 'min']

    # plots ROC curve for each metric
    for x, name in zip(x_list, x_names):
        fpr, tpr, thresh = stat.thresh_roc(x, y)
        print('\n', name)
        ax = stat.plot_roc(fpr, tpr, show_fig=True)
    
    # closes plots to save memory
    plt.close('all')
    
    return bkgd



######################### DEMO 3 - HOW TO SEGMENT OBJECTS ##############################

def compare_thresh(vid_path, num, thresh, th_lo, th_hi, bkgd=None,
                   num_frames_for_bkgd=100, x_range=None, y_range=None,
                   scale_fig=1, scale_height=1.4, brightness=1, mag=4, 
                   show_fig=True, shape=None, t_fs=18):
    """
    Compares standard and hysteresis thresholds on an image.
    Shows linked interactive Bokeh images of the different figures.
    Shows 4 images:
    1) original image
    2) background-subtracted image
    3) uniformly threshold applied to #2
    4) hysteresis threshold applied to #2
    
    Parameters
    ----------
    vid_path : string
        Filepath to video to analyze
    num : int
        Number of frame to analyze (0-indexed)
    thresh : int
        Uniform threshold to apply
    th_lo, th_hi : int
        Low and high thresholds for hysteresis thresholding
    bkgd : (M x N) numpy array of uint8s, opt (default=None)
        Background to subtract from each frame; if None provided,
        computes background by computing the median of requested number
        of frames
    num_frames_for_bkgd : int, opt (default=100)
        If `bkgd` is `None`, computes background by taking the median of
        this many frames, starting with the first frame
    x_range, y_range : 2-tuple of ints, opt (default=None)
        Range of (x-)/(y-)values to show in Bokeh plot; if None, shows entire image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    scale_height : float, opt (default=1.4)
        Factor by which to scale height of image after applying 
        initial scaling (scale_fig)
    brightness : float, opt (default=1)
        Factor by which to scale brightness of image--too bright and the image
        will saturate
    mag : int, opt (default=4)
        Magnification of the objective of the microscope
    show_fig : bool, opt (default=True)
        If True, figures showing the thresholded frame will be shown
    shape : 2-tuple of ints, opt (default=None)
        Number of rows and columns (#rows, #cols) in which to display the 
        4 images; if None, displays in a column
    t_fs : int, opt (default=18)
        Fontsize for the title of each image
        
    Returns
    -------
    p_grid : Bokeh figure
        Figure of 4 images described above (linked interactivity)
    bkgd : (M x N) numpy array of uint8s
        Background used for background subtraction
    frame_gray : (M x N) numpy array of uint8s
        Black-and-white version of requested frame from video
    im_diff : (M x N) numpy array of ints
        Background subtracted frame requested from video (preserves sign)
    im_thresh : (M x N) numpy array of uint8s
        `im_diff` with uniform threshold applied (0 for below threshold, 255 for above)
    im_thresh_hyst : (M x N) numpy array of uint8s
        Same as `im_thresh` but hysteresis threshold is applied
    """
    ### BACKGROUND SUBTRACTION ###
    signed_diff = improc.bkgd_sub_signed(vid_path, num, bkgd=bkgd, 
                                         num_frames_for_bkgd=num_frames_for_bkgd)
    # sets positive differences to zero and then takes absolute value
    signed_diff[signed_diff > 0] = 0
    im_diff = np.abs(signed_diff).astype('uint8')
    
    ### THRESHOLDING ###
    # thresholds image to become black-and-white
    im_thresh = improc.thresh_im(im_diff, thresh)
    
    # apply hysteresis threshold
    im_thresh_hyst = improc.hysteresis_threshold(im_diff, th_lo, th_hi)

    # loads pixel to micron conversion based on objective magnification
    pix_per_um = pix_per_um_photron[mag]
    
    # plots grid of 4 linked images for comparison
    frame, _ = basic.load_frame(vid_path, num)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_list = [frame_gray, im_diff, im_thresh, im_thresh_hyst]
    title_list = ['Frame {0:d}'.format(num), 'Background Subtracted', 'Uniform Threshold', 'Hysteresis Threshold']
    if shape is None:
        shape = (len(frame_list),1)
    p_grid = pltim.linked_ims(frame_list, pix_per_um, scale_fig=scale_fig, shape=shape, 
                              x_range=x_range, y_range=y_range, title_list=title_list, show_fig=show_fig, 
                              brightness=brightness, palette='Greys256', scale_height=scale_height, t_fs=t_fs)
    
    return p_grid, bkgd, frame_gray, im_diff, im_thresh, im_thresh_hyst


def compare_closing(im_thresh, r_selem, x_range=None, scale_fig=1, mag=4):
    """
    Compares original thresholded image with dilated image and 
    dilated + eroded (=closed) image.
    
    Parameters
    ----------
    im_thresh : (M x N) numpy array of uint8s
        Thresholded image (0s and 255s)
    r_selem : int
        radius of structuring element [pixels]
    x_range : 2-tuple of ints, opt (default=None)
        Range of x-values to show in figure; if `None`, shows whole width of image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    mag : int, opt (default=4)
        Magnification of objective lens of microscope
        
    Returns
    -------
    p_grid : Bokeh figure
        Linked frames showing thresholded image, dilated image, and closed image
    dilated : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after applying dilation
    closed : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after applying closing
    """
    # creates structuring element (disk shape)
    selem = skimage.morphology.selem.disk(r_selem)

    # dilates and erodes image (= closes)
    dilated = skimage.morphology.binary_dilation(im_thresh, selem=selem)
    closed = skimage.morphology.binary_erosion(dilated, selem=selem)
    
    # plots thresholded, dilated, and closed images in linked frames
    im_list = [im_thresh, dilated, closed]
    title_list = ['Thresholded', 'Dilated', 'Closed']
    p_grid = pltim.linked_ims(im_list, pix_per_um_photron[mag], shape=(3,1), palette='Greys256',
                         scale_fig=scale_fig, title_list=title_list, x_range=x_range)

    return p_grid, dilated, closed


def compare_opening(im_thresh, r_selem, x_range=None, scale_fig=1, mag=4):
    """
    Compares original thresholded image with eroded image and 
    eroded + dilated (=opened) image.
    
    Parameters
    ----------
    im_thresh : (M x N) numpy array of uint8s
        Thresholded image (0s and 255s)
    r_selem : int
        radius of structuring element [pixels]
    x_range : 2-tuple of ints, opt (default=None)
        Range of x-values to show in figure; if `None`, shows whole width of image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    mag : int, opt (default=4)
        Magnification of objective lens of microscope
        
    Returns
    -------
    p_grid : Bokeh figure
        Linked frames showing thresholded image, eroded image, and opened image
    eroded : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after applying erosion
    opened : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after applying opening
    """
    # creates structuring element (disk shape)
    selem = skimage.morphology.selem.disk(r_selem)

    # dilates and erodes image (= closes)
    eroded = skimage.morphology.binary_erosion(im_thresh, selem=selem)
    opened = skimage.morphology.binary_dilation(eroded, selem=selem)
    
    # plots thresholded, dilated, and closed images in linked frames
    im_list = [im_thresh, eroded, opened]
    title_list = ['Thresholded', 'Eroded', 'Opened']
    p_grid = pltim.linked_ims(im_list, pix_per_um_photron[mag], shape=(3,1), palette='Greys256',
                         scale_fig=scale_fig, title_list=title_list, x_range=x_range)

    return p_grid, eroded, opened
    
    
def compare_remove_small_objects(im_thresh, min_size, x_range=None, scale_fig=1, mag=4):
    """
    Shows effect of removing small objects on thresholded image.
    
    Parameters
    ----------
    im_thresh : (M x N) numpy array of uint8s
        Thresholded image (0s and 255s)
    min_size : int
        Minimum number of pixels a contiguous object must have to be kept
    x_range : 2-tuple of ints, opt (default=None)
        Range of x-values to show in figure; if `None`, shows whole width of image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    mag : int, opt (default=4)
        Magnification of objective lens of microscope
        
    Returns
    -------
    p_grid : Bokeh figure
        Linked frames showing thresholded image and image with small objects removed
    small_obj_rm : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after removing small objects
    """
    # removes small objects from thresholded image
    small_obj_rm = skimage.morphology.remove_small_objects(im_thresh.astype(bool),
                                                           min_size=min_size)
    
    # plots thresholded, dilated, and closed images in linked frames
    im_list = [im_thresh, small_obj_rm]
    title_list = ['Thresholded', 'Small Objects Removed']
    p_grid = pltim.linked_ims(im_list, pix_per_um_photron[mag], shape=(2,1), palette='Greys256',
                         scale_fig=scale_fig, title_list=title_list, x_range=x_range)

    return p_grid, small_obj_rm
    
    
def compare_fill_holes(im_thresh, x_range=None, scale_fig=1, mag=4):
    """
    Compares effect of filling holes in a thresholded image.
    
    Parameters
    ----------
    im_thresh : (M x N) numpy array of uint8s
        Thresholded image (0s and 255s)
    min_size : int
        Minimum number of pixels a contiguous object must have to be kept
    x_range : 2-tuple of ints, opt (default=None)
        Range of x-values to show in figure; if `None`, shows whole width of image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    mag : int, opt (default=4)
        Magnification of objective lens of microscope
        
    Returns
    -------
    p_grid : Bokeh figure
        Linked frames showing thresholded image and an image with the holes filled
    filled_holes : (M x N) numpy array of uint8s
        0s and 255s showing thresholded image after filling holes
    """
    # fills holes in thresholded image
    filled_holes = improc.frame_and_fill(im_thresh)
    
    # plots thresholded, dilated, and closed images in linked frames
    im_list = [im_thresh, filled_holes]
    title_list = ['Thresholded', 'Filled Holes']
    p_grid = pltim.linked_ims(im_list, pix_per_um_photron[mag], shape=(2,1), palette='Greys256',
                         scale_fig=scale_fig, title_list=title_list, x_range=x_range)

    return p_grid, filled_holes


def segment_and_measure(vid_path, num, th_lo, th_hi, r_selem, min_size, 
                        width_border=2, bkgd=None, num_frames_for_bkgd=100,
                        x_range=None, y_range=None, scale_fig=1, scale_height=1.4,
                        brightness=1, mag=4, show_fig=True):
    """
    Segments frame from video using the techniques demonstrated in the previous
    functions. Then measures the properties of each segmented object and plots
    some of those properties (major and minor axes, bounding box, etc.) on the image.
    
    Parameters
    ----------
    vid_path : string
        Filepath to video to analyze
    num : int
        Number of frame in video to analyze (0-indexed)
    th_lo, th_hi : int
        Low and high thresholds used for hysteresis thresholding
    r_selem : int
        Radius of structuring element
    min_size : int
        Minimum number of pixels a contiguous object must have to be kept
    width_border : int, opt (default=2)
        Number of pixels considered to be part of "border"--objects touching
        these pixels will be considered "on border" and will have holes that
        open onto this border filled in
    bkgd : (M x N) numpy array of uint8s, opt (default=None)
        If provided, this background will be subtracted from the frame.
        If `None`, background will be computed by taking the median of the
        first `num_frames_for_bkgd`.
    num_frames_for_bkgd : int, opt (default=100)
        If `bkgd` is `None`, computes background by taking the median of the
        first `num_frames_for_bkgd`.
    x_range, y_range : 2-tuple of ints, opt (default=None)
        Range of (x-)/(y-)values to show in Bokeh plot; if None, shows entire image
    scale_fig : float, opt (default=1)
        Factor by which to scale the size of the figure
    scale_height : float, opt (default=1.4)
        Factor by which to scale height of image after applying 
        initial scaling (scale_fig)
    brightness : float, opt (default=1)
        Factor by which to scale brightness of image--too bright and the image
        will saturate
    mag : int, opt (default=4)
        Magnification of the objective of the microscope
    show_fig : bool, opt (default=True)
        If True, figures showing the thresholded frame will be shown
        
    Returns
    -------
    p_grid : Bokeh figure
        Linked frames showing original image and segmented image with properties plotted
    objects : dictionary
        Dictionary of region properties of each object, labeled by index
    """
    # background subtraction (median, threaded)
    signed_diff = improc.bkgd_sub_signed(vid_path, num, bkgd=bkgd, 
                                         num_frames_for_bkgd=num_frames_for_bkgd)

    # sets positive differences to zero and then takes absolute value
    signed_diff[signed_diff > 0] = 0
    im_diff = np.abs(signed_diff).astype('uint8')

    # apply hysteresis threshold
    im_thresh_hyst = improc.hysteresis_threshold(im_diff, th_lo, th_hi)

    # fill holes
    filled_holes = improc.frame_and_fill(im_thresh_hyst)
    
    # creates structuring element (disk shape)
    selem = skimage.morphology.selem.disk(r_selem)
    # erodes and dilates image (= opens)
    opened = skimage.morphology.binary_opening(filled_holes, selem=selem)
    # dilates and erodes image (= closes)
    closed = skimage.morphology.binary_closing(opened, selem=selem)

    # removes small objects from thresholded image
    small_obj_rm = skimage.morphology.remove_small_objects(closed, min_size=min_size)

    # finds objects and assigns IDs to track them, saving to archive
    frame_labeled = skimage.measure.label(small_obj_rm)
    
    # creates dictionaries of properties for each object
    objects = dh.store_region_props(frame_labeled, num, width_border=width_border)

    # looks up pixels per micron conversion
    pix_per_um = pix_per_um_photron[mag]
    
    # plots objects with major and minor axes, convex hull, etc.
    # first, color objects by label--must convert back to uint8 afterwards (cvify)
    image_label_overlay = basic.cvify(skimage.color.label2rgb(frame_labeled, image=filled_holes, bg_label=0))
    # gets original frame to show in background
    frame, _ = basic.load_frame(vid_path, num)
    # show images in bokeh  
    p_im = pltim.format_im(frame, pix_per_um, x_range=x_range, y_range=y_range,
                        scale_fig=scale_fig, scale_height=scale_height, show_fig=False,
                       title='Frame {0:d}'.format(num), palette=None)
    p = pltim.format_im(image_label_overlay, pix_per_um, x_range=x_range, y_range=y_range, 
                        scale_fig=scale_fig, scale_height=scale_height, 
                        brightness=brightness, palette='Colorblind8', show_fig=False,
                        title='Region Props of Frame {0:d}'.format(num), hide_colorbar=True)
   
    # graphically overlays measured properties
    p = dh.label_and_measure_objs(p, frame_labeled, objects, pix_per_um)
    
    p_grid = pltim.make_gridplot([p_im, p], (2,1))
    if show_fig:
        show(p_grid)

    return p_grid, objects
    
    
################### DEMO 4 - HOW TO TRACK OBJECTS ###########################

def ims_to_vid(im_dir, output_filepath, start, end, every):
    """
    Converts images to video.
    
    im_dir : string
        Directory containing images to be converted into a video
    output_filepath : string
        Filepath to which to save video, including video name and extension (mp4 or avi)
    start : int
        Start loading images from this number in the list (0-indexed)
    end : int
        Stop loading images from this number in the list (0-indexed)
    every : int
        Load one image every this number of images (so skip every-1 images in b/w)
        
    Returns
    -------
    void
    
    Sources
    1. https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    Copied code for loading filenames of all images
    """
    # loads filenames of all images
    # from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    _, _, filenames_unsorted = next(os.walk(im_dir))

    filenames = sorted(filenames_unsorted)
    if end >= 0:
        filenames = filenames[start:end:every]
    else:
        filenames = filenames[start::every]
    
    ext = output_filepath[-3:]
    if ext =='mp4' or ext == 'avi':
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'H264')

        # initializes videowriter
        writer = None

        for filename in filenames:
            frame = cv2.imread(os.path.join(im_dir, filename))
            # initializes video writer the first time through the loop
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_filepath, fourcc, fps,
                        (w, h), True)
            try:
                writer.write(frame)
                print('wrote frame {0:d}'.format(i))
            except:
                print('Failed to write frame.')

        writer.release()
        
    elif ext == 'gif':
        images = []
        for filename in filenames:
            images.append(imageio.imread(os.path.join(im_dir, filename)))
        imageio.mimsave(output_filepath, images)
        
    return



def save_tracked_ims(vid_path, objs, frame_IDs, save_dir, start=0, end=-1, every=1, 
                     ext='jpg', color=(0,255,0), offset=10, label_key=None):
    """
    Saves images with centroid, bounding box, and label overlaid on objects
    after computing them with a tracking method like `track_objects()`.
    
    Saved images can be converted into a video/GIF using `ims_to_vid()`.
    
    Parameters
    ----------
    vid_path : string
        Filepath to video to analyze
    objs : dictionary
        Dictionary indexed by ID # of each object containing region properties
        and others of the object
    frame_IDs : dictionary
        Dictionary indexed by frame # contain a list of the ID #s of the
        objects detected in the given frame
    save_dir : string
        Pathway to the directory in which to save the tracked images
    start : int, opt (default=0)
        Start saving frames with this number (0-indexed)
    end : int, opt (default=-1)
        Stopping saving images at this number (if -1, saves up to and 
        including the last frame)
    every : int, opt (default=1)
        Saves every this many frames (or skips every-1 frames in b/w saved frames)
    ext : string, opt (default='jpg')
        Extension for image format, excluding '.'
    color : 3-tuple of uint8s, opt (default=(0,255,0))
        Color of annotations in BGR; default is green
    offset : int, opt (default=10)
        number of pixels to offset label in the x-direction (to the right)
    label_key : dictionary, opt (default=None)
        key converting ID #s into more meaningful labels (key = ID #, value = label)
        If `None`, just shows ID #s as labels
        
    Returns
    -------
    void
    """
    # makes save directory if it does not exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # loads video
    cap = cv2.VideoCapture(vid_path)
    # chooses end frame to be last frame if given as -1
    if end == -1:
        end = basic.count_frames(vid_path)

    # loops through frames of video with objects according to data file
    for f in range(start, end, every):
        frame = basic.read_frame(cap, f)

         # labels objects
        IDs = frame_IDs[f]
        for ID in IDs: 
            dh.bbox_obj(frame, objs, ID, f, color, label_key=label_key, offset=offset)

        # save
        im = PIL.Image.fromarray(frame)
        im.save(os.path.join(save_dir, '{0:03d}.{1:s}'.format(f, ext)))

    return

    
def track_objects(vid_path, th, th_lo, th_hi, min_size,
                  r_selem, start=0, end=-1, every=1, print_freq=10, 
                  width_border=2, bkgd=None, num_frames_for_bkgd=100,  
                  remember_objs=False, n_consec=3):
    """
    Tracks objects in the given video and saves data about those objects in
    a pickle file. Apply to generic video with high frame rate relative to motion--
    tracks objects by associating them with the nearest object in the subsequent frame.
    
    Results can be used to annotate images with `save_tracked_ims()`.
    
    Parameters
    ----------
    vid_path : string
        Filepath to video to analyze
    th_lo, th_hi : int
        Low and high thresholds used for hysteresis thresholding
    min_size : int
        Minimum number of pixels a contiguous object must have to be kept
    r_selem : int
        Radius of structuring element
    width_border : int, opt (default=2)
        Number of pixels considered to be part of "border"--objects touching
        these pixels will be considered "on border" and will have holes that
        open onto this border filled in
    bkgd : (M x N) numpy array of uint8s, opt (default=None)
        If provided, this background will be subtracted from the frame.
        If `None`, background will be computed by taking the median of the
        first `num_frames_for_bkgd`.
    num_frames_for_bkgd : int, opt (default=100)
        If `bkgd` is `None`, computes background by taking the median of the
        first `num_frames_for_bkgd`.
    remember_objs : bool, opt (default=False)
        If True, asks assignment algorithm to remember objects that are not observed 
        in the current frame) and to predict their centroid based on their average velocity. 
    n_consec : int, opt (default=3)
        Number of consecutive frames in which an object must appear to be considered
        a "true" object and not be filtered out
        
    Returns
    -------
    objs_filtered : dictionary
        Dictionary of objects that passed the filter, indexed by ID # and containing
        properties in the format of a TrackedObject (see `TrackedObject` in `classes/classes.py`)
    frame_IDs : dictionary
        Dictionary of frame numbers with lists of objects detect in that frame; includes
        objects that were filtered out and not included in `objs_filtered`
    """    
    # computes background with median filtering
    if bkgd is None:
        bkgd = improc.compute_bkgd_med_thread(vid_path,
                        vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
                        num_frames=num_frames_for_bkgd)
        
    if end == -1:
        end = basic.count_frames(vid_path)

    # organizes arguments for object segmentation function
    track_kwargs = {'vid_path' : vid_path,
        'bkgd' : bkgd,
        'print_freq' : print_freq,
        'start' : start,
        'end' : end,
        'every' : every,
        'flow_dir' : (0,0),
        'assign_objects_method' : improc.assign_objs
    }

    # all arguments for highlight_object_method besides "frame" and "bkgd" (first 2)
    highlight_kwargs = {'th' : th,
                        'th_lo' : th_lo,
                        'th_hi' : th_hi,
                        'min_size_hyst' : min_size,
                        'min_size_th' : min_size,
                        'width_border' : width_border,
                        'selem' : cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_selem, r_selem)),
    }

    assignobjects_kwargs = {'d_fn' : improc.d_euclid_bw_obj,
                            'd_fn_kwargs' : {},
                            'fps' : cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS),
                            'row_lo' : 0, # doesn't crop if both are 0
                            'row_hi' : 0, # doesn't crop if both are 0
                            'remember_objs' : remember_objs
    }

    # times and executes object-tracking
    start_time = time.time()
    objs, frame_IDs = improc.track_object(improc.track_object_cvvidproc,
        track_kwargs, highlight_kwargs, assignobjects_kwargs, ret_IDs=True)
    print('{0:d} frames analyzed with object-tracking in {1:.3f} s.'.format(
        int((end-start)/every),
        time.time()-start_time))
    
    # removes objects that don't appear in consecutive frames
    objs_filtered = {}
    for i in objs:
        if improc.appears_in_consec_frames(objs[i], n_consec):
            objs_filtered[i] = objs[i]
    
    return objs_filtered, frame_IDs



######################## DEMO 5 - PROCESSING A FULL BUBBLE GROWTH VIDEO #########################

def print_flow_props(p):
    """
    Prints some estimated flow properties.
    
    Parameters
    ----------
    p : dictionary
        Dictionary of parameters of video processing
    
    Returns
    -------
    dp : float
        Predicted pressure drop down observation capillary [Pa]
    v_max : float
        Predicted velocity at the center of the inner stream [m/s]
    R_i : float
        Predicted inner stream radius [m]
    """
    # computes pressure drop down channel
    vid_params = fn.parse_vid_path(p['vid_name'])
    Q_i = uLmin_2_m3s * vid_params['Q_i']
    Q_o = uLmin_2_m3s * vid_params['Q_o']
    dp, R_i, v_max = flow.get_dp_R_i_v_max(p['eta_i'], p['eta_o'], p['L'], Q_i, Q_o, p['R_o'], SI=True)

    print('------Estimated Flow Properties------')
    print('Pressure drop = {0:.1f} bar'.format(dp*Pa_2_bar))
    print('Expected velocity = {0:.1f} m/s'.format(v_max))
    print('Expected inner stream width = {0:.1f} um\n'.format(2*R_i*m_2_um))
    
    return dp, v_max, R_i


def process_video(p, replace=False, use_prev_bkgd=False, 
                  mask_data_filepath=None, print_freq=100,
                 compare_to_py=False, max_frames_py=1000):
    """
    Segments and tracks objects in a video of bubble growth in which
    bubbles travel from left to right at a roughly predictable speed.
    
    Parameters
    ----------
    p : dictionary
        Parameters for the image processing
    replace : bool, opt (default=False)
        If True, will replace existing data if saved under the same name
    use_prev_bkgd : bool, opt (default=False)
        If True, will try to load existing data to use its background, 
        rather than computing the background itself
    mask_data_filepath : string, opt (default=None)
        If provided, will automatically use the mask data at the given filepath (.pkl)
    print_freq : int, opt (default=100)
        Legacy parameter--may affect some versions of CvVidProc
    compare_to_py : bool, opt (default=False)
        If True, will compare timing of C++-driven, multithreaded CvVidProc
        image processing with a version written purely in Python
    max_frames_py : int, opt (default=1000)
        Because the Python image processing code is so slow, this parameter
        limits the number of frames that it is allowed to analyze
        
    Returns
    -------
    bubbles : dictionary
        Data about the bubbles/objects segmented and tracked, indexed by ID #
    frame_IDs : dictionary
        Indexed by frame number, each entry is a list of the bubble ID #s seen
        in that frame, which can assist with faster searches in some cases
    """
    # defines filepath to video
    vid_path, vid_dir, data_dir, figs_dir = dh.get_dirs(p)
    
    # adjusts a few input parameters
    p['selem'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['selem_dim'], p['selem_dim']))
    p['fps'] = fn.parse_vid_path(vid_path)['fps']
    # if last frame given as -1, returns as final frame of video
    if p['end'] == -1:
        p['end'] = basic.count_frames(vid_path)
    # checks that video has the requested frames
    # subtracts 1 since "end" gives an exclusive upper bound [start, end)
    if not basic.check_frames(vid_path, p['end']-1):
        print('Terminating analysis. Please enter valid frame range next time.')
        return

    # extracts parameters recorded in video name
    vid_params = fn.parse_vid_path(vid_path)
    Q_i = uLmin_2_m3s*vid_params['Q_i'] # inner stream flow rate [m^3/s]
    Q_o = uLmin_2_m3s*vid_params['Q_o'] # outer stream flow rate [m^3/s]
    mag = vid_params['mag'] # magnification of objective (used for pix->um conv)
    # gets unit conversion for Photron camera based on microscope magnification
    if p['photron']:
        pix_per_um = pix_per_um_photron[mag]
    # otherwise uses conversion for Chronos camera
    else:
        pix_per_um = pix_per_um_chronos[mag]

    # creates directories recursively if they do not exist
    fn.makedirs_safe(data_dir)
    fn.makedirs_safe(figs_dir)
    # defines name of data file to save
    data_path = os.path.join(data_dir,'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'],
                                                        p['every'], p['end']))
    # if the data file already exists in the given directory, end analysis
    # upon request
    # if replacement requested, previous background may be used upon request
    if os.path.isfile(data_path):
        if not replace:
            print('{0:s} already exists. Terminating analysis.'.format(data_path))
            return

    # loads data for masking image--assumes standard file name format if not provided
    if mask_data_filepath == None:
        # standard file name format is video name + '_mask.pkl'; [:-4] removes extension
        mask_data_filepath = vid_path[:-4] + '_mask.pkl'
    with open(mask_data_filepath, 'rb') as f:
        mask_data = pkl.load(f)
    # computes minimum and maximum rows for bubble tracking computation
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)

    # gets background for image subtraction later on
    data_existing = glob.glob(os.path.join(data_dir, 'f_*.pkl'))
    if len(data_existing) > 0 and use_prev_bkgd:
        # loads background from first existing data file
        with open(data_existing[0], 'rb') as f:
            data_prev = pkl.load(f)
        bkgd = data_prev['metadata']['bkgd']
    else:
        # computes background with median filtering
        bkgd = improc.compute_bkgd_med_thread(vid_path,
            vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
            num_frames=p['num_frames_for_bkgd'],
            crop_y=row_lo,
            crop_height=row_hi-row_lo)

    # computes pressure drop [Pa], inner stream radius [m], and max velocity
    #  [m/s] for Poiseuille sheath flow
    dp, R_i, v_max = flow.get_dp_R_i_v_max(p['eta_i'], p['eta_o'], p['L'],
                                        Q_i, Q_o, p['R_o'], SI=True)

    ######################## 2) TRACK BUBBLES ##################################
    # organizes arguments for bubble segmentation function
    
    track_kwargs = {'vid_path' : vid_path,
        'bkgd' : bkgd,
        'highlight_object_method' : cfg.highlight_methods[p['highlight_method']],
        'print_freq' : print_freq,
        'start' : p['start'],
        'end' : p['end'],
        'every' : p['every'],
        'assign_objects_method' : improc.assign_objects
    }

    highlight_kwargs = {'th' : p['th'],
        'th_lo' : p['th_lo'],
        'th_hi' : p['th_hi'],
        'min_size_hyst' : p['min_size_hyst'],
        'min_size_th' : p['min_size_th'],
        'width_border' : p['width_border'],
        'selem' : p['selem'],
        'mask_data' : mask_data
    }

    # defines keyword arguments for distance function used in tracking
    d_fn_kwargs = {
        'axis' : flow_dir,
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'v_max' : v_max*m_2_um*pix_per_um,  # convert max velocity from [m/s] to [pix/s] first
        'fps' : p['fps'],
    }
    # additional variables required for method used to assign labels to objects
    assign_kwargs = {
        'fps' : p['fps'],  # extracts fps from video filepath
        'd_fn' : improc.bubble_distance_v,
        'd_fn_kwargs' : d_fn_kwargs,
        'width_border' : p['width_border'],
        'min_size_reg' : p['min_size_reg'],
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'remember_objects' : False, # TODO -- make this a parameter that can be changed by user
        'object_kwargs' : {'flow_dir' : flow_dir, 'pix_per_um' : pix_per_um},
    } 


    # parallelized C++ tracking (fast)
    # number of frames analyzed
    n_frames_cv = int((p['end']-p['start'])/p['every'])
    # starts timer
    start_time = time.time()
    # tracks bubbles
    bubbles, frame_IDs = improc.track_object(improc.track_object_cvvidproc,
        track_kwargs, highlight_kwargs, assign_kwargs, ret_IDs=True)
    # stops timer
    time_cv = time.time()-start_time
    
    print('{0:d} frames analyzed with cvvidproc (C++, parallelized) bubble-tracking in {1:.3f} s.'.format(n_frames_cv, time_cv))
          
    
    # pure python tracking (slow)
    if compare_to_py:
        # limits number of frames to analyze with this slow method
        track_kwargs_py = track_kwargs
        end_py = min(p['end'], max_frames_py + p['start'])
        track_kwargs_py['end'] = end_py
        # total number of frames analyzed
        n_frames_py = int((end_py-p['start'])/p['every'])
        # announces comparison
        print('\n------------------------------------------------------------------')
        print('Compare to pure Python (comparison only--data not saved):')
        # starts timer
        start_time = time.time()
        # tracks bubbles
        _ = improc.track_object(improc.track_object_py,
            track_kwargs_py, highlight_kwargs, assignobjects_kwargs, ret_IDs=True)
        # stops timer
        time_py = time.time()-start_time
          
        print('{0:d} frames analyzed with pure Python bubble-tracking in {1:.3f} s.'.format(n_frames_py, time_py))
        print('------------------------------------------------------------------\n')

        # computes speed-up with cvvidproc
        rate_cv = n_frames_cv / time_cv
        rate_py = n_frames_py / time_py
        speedup_cv = rate_cv / rate_py
        print('Parallelizing and using C++ speeds up tracking by',
              '***{0:d}x*** compared to pure Python. Thanks Isaac!\n\n'.format(int(speedup_cv)))
        
        
    ######################## 3) PROCESS DATA ###################################
    print('Processing data...')
    # computes velocity at interface of inner stream [m/s]
    v_inner = flow.v_inner(Q_i, Q_o, p['eta_i'], p['eta_o'], p['R_o'], p['L'])
    for ID in bubbles.keys():
        bubble = bubbles[ID]
        # computes average speed [m/s]
        bubble.proc_props()

    ########################## 4) SAVE RESULTS #################################
    print('Saving results...')
    # stores metadata (I will not store video parameters or parameters from the
    # input file since they are stored elsewhere already)
    metadata = {'input_name' : p['input_name'], 'bkgd' : bkgd, 'flow_dir' : p['flow_dir'],
                'mask_data' : mask_data, 'row_lo' : row_lo, 'row_hi' : row_hi,
                'dp' : dp, 'R_i' : R_i, 'v_max' : v_max, 'v_inner' : v_inner,
                'args' : highlight_kwargs, 'frame_IDs' : frame_IDs,
                'pix_per_um' : pix_per_um, 'input_params' : p}

    # stores data
    data = {}
    data['objects'] = bubbles
    data['frame IDs'] = frame_IDs
    data['metadata'] = metadata

    # also saves copy of mask file
    print('Copying mask file to {0:s}'.format(data_dir))
    shutil.copyfile(vid_path[:-4] + '_mask.pkl',
                        os.path.join(data_dir, 'mask.pkl'))

    # saves input data
    input_data_path = os.path.join(data_dir, 'input.pkl')
    with open(input_data_path, 'wb') as f:
        print('saving input data in {0:s}'.format(input_data_path))
        pkl.dump(p, f)
    # saves output data
    with open(data_path, 'wb') as f:
        print('saving output data in {0:s}'.format(data_path))
        pkl.dump(data, f)
        
    return bubbles, frame_IDs


def highlight_obj(p, skip_blanks=True, ext='jpg', color_object=True, 
                  brightness=1.0, offset=5, quiet=True):
    """
    Based on highlight.py in bubbletracking_koe/analysis. Saves frames with
    colored and labeled objects after processing a video with `process_video()`.
    
    p : dictionary
        Parameter for the image processing (same as used in process_video())
    skip_blanks : bool, opt (default=True)
        If True, does not process or save frames without any objects in them
    ext : string, opt (default='jpg')
        Extension for image files--excludes '.'
    color_object : bool, opt (default=True)
        If True, shades in region segmented to belong to a object
    brightness : float, opt (default=1.0)
        Factor by which to scale original brightness of video
    offset : int, opt (default=5)
        Offset of labels of objects. If 0, the lower left will be at the centroid.
        Increasing the offset shifts the label to the right (larger x).
    quiet : bool, opt (default=True)
        If False, prints out line for each frame saved. If True, no printouts.
        
    Returns
    -------
    void
    """
    vid_path, vid_dir, data_dir, figs_dir = dh.get_dirs(p)
    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'], 
                                                    p['every'], p['end']))
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
    # chooses end frame to be last frame if given as -1
    if p['end'] == -1:
        p['end'] = basic.count_frames(vid_path)

    # loops through frames of video with bubbles according to data file
    for f in range(p['start'], p['end'], p['every']):

        # skips frame upon request if no bubbles identified in frame
        if len(frame_IDs[f]) == 0 and skip_blanks:
            continue
        # loads frame
        frame = basic.read_frame(cap, f)
        # crops frame
        row_lo = metadata['row lo']
        row_hi = metadata['row hi']
        frame = frame[row_lo:row_hi, :]
        # extracts value channel
        val = basic.get_val_channel(frame)

        # highlights bubble according to parameters from data file
        bkgd = metadata['bkgd']
        bubble = cfg.highlight_methods[p['highlight_method']](val, bkgd, **metadata['args'])
        # applies highlights
        frame_labeled, num_labels = skimage.measure.label(bubble, return_num=True)
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
            outer_stream = bubbles[ID].get_props('inner stream') == 0
            error = bubbles[ID].get_props('inner stream') == -1
            if on_border or outer_stream:
                color = cfg.black
            elif not error:
                color = cfg.white
            else:
                color = cfg.red
            frame_disp = cv2.putText(img=frame_disp, text=str(ID), org=(x+offset, y),
                                    fontFace=0, fontScale=0.5, color=color,
                                    thickness=2)

        # saves image
        im = PIL.Image.fromarray(frame_disp)
        im.save(os.path.join(figs_dir, '{0:d}.{1:s}'.format(f, ext)))

        if not quiet:
            print('Saved frame {0:d} in {1:d}:{2:d}:{3:d}.'.format(f, 
                                             p['start'], p['every'], p['end']))
            
        return

