# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:31:17 2020

improc-analytics.py contains functions used to analyze image-processing methods
defined in improc.py.

@author: Andy
"""

# imports standard libs
import numpy as np
import cv2

# imports bokeh modules
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot

# imports custom libraries
import cvimproc.basic as basic


def bokehfy(im, vert_flip=True):
    """
    Formats image for display with Bokeh. Can accommodate boolean, 0-1 scaled
    floats, and 0-255 scaled uint8 images.

    Parameters
    ----------
    im : numpy array
        Image to be formatted for Bokeh
    vert_flip : bool
        Flag indicating if the image should be flipped using cv2.flip().
        Used because Bokeh inherently flips objects, so flipping them before-
        hand cancels the effect.

    Returns
    -------
    im : numpy array
        Original image formatted for display with Bokeh.
    """
    # converts boolean images to uint8
    if im.dtype == 'bool':
        im = 255*im.astype('uint8')
    # converts 0-1 scale float to 0-255 scale uint8
    elif im.dtype == 'float':
        # multiplies by 255 if scaled by 1
        if np.max(im) <= 1.0:
            im *= 255
        # converts to uint8
        im = im.astype('uint8')
    # converts BGR images to RGBA (from CV2, which uses BGR instead of RGB)
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
    # converts gray scale (2d) images to RGBA
    elif len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
    if vert_flip:
        im = cv2.flip(im, 0) # because Bokeh flips vertically

    return im


def format_frame(frame, pix_per_um, fig_size_red, brightness=1.0, title=None):
    frame = basic.adjust_brightness(frame, brightness)
    width = frame.shape[1]
    height= frame.shape[0]
    width_um = int(width / pix_per_um)
    height_um = int(height / pix_per_um)
    width_fig = int(width*fig_size_red)
    height_fig = int(height*fig_size_red)
    p = figure(x_range=(0,width_um), y_range=(0,height_um), output_backend="webgl",
               width=width_fig, height=height_fig, title=title)
    p.xaxis.axis_label = 'width [um]'
    p.yaxis.axis_label = 'height [um]'
    im = p.image_rgba(image=[frame], x=0, y=0, dw=width_um, dh=height_um)

    return p, im


def linked_four_frames(four_frames, pix_per_um, fig_size_red, show_fig=True):
    """
    Shows four frames with linked panning and zooming.
    """
    # list of figures
    p = []
    # creates images
    for frame in four_frames:
        p_new, _ = format_frame(frame, pix_per_um, fig_size_red)
        p += [p_new]
    # sets ranges
    for i in range(1, len(p)):
        p[i].x_range = p[0].x_range # links horizontal panning
        p[i].y_range = p[0].y_range # links vertical panning
    # creates gridplot
    p_grid = gridplot([[p[0], p[1]], [p[2], p[3]]])
    # shows figure
    if show_fig:
        show(p_grid)

    return p_grid


def linked_frames(frame_list, pix_per_um, fig_size_red, shape=(2,2),
                  show_fig=True, brightness=1.0, title_list=[]):
    """
    Shows multiple frames with linked panning and zooming.
    """
    # list of figures
    p = []
    # creates images
    for frame in frame_list:
        p_new, _ = format_frame(frame, pix_per_um, fig_size_red, brightness=brightness)
        p += [p_new]
    # adds titles to plots if provided (with help from
    # https://stackoverflow.com/questions/47733953/set-title-of-a-python-bokeh-plot-figure-from-outside-of-the-figure-functio)
    for i in range(len(title_list)):
        t = Title()
        t.text = title_list[i]
        p[i].title = t
    # sets ranges
    for i in range(1, len(p)):
        p[i].x_range = p[0].x_range # links horizontal panning
        p[i].y_range = p[0].y_range # links vertical panning
    # formats list for gridplot
    n = 0
    p_table = []
    for r in range(shape[0]):
        p_row = []
        for c in range(shape[1]):
            if n >= len(p):
                break
            p_row += [p[n]]
            n += 1
        p_table += [p_row]
        if n >= len(p):
            break

    # creates gridplot
    p_grid = gridplot(p_table)
    # shows figure
    if show_fig:
        show(p_grid)

    return p_grid


def six_frame_eda(vid_filepath, f, params, highlight_method, pix_per_um,
                  fig_size_red, tag=''):
    """Shows six steps in the image-processing of a frame."""
    # loads current frame for image subtraction
    frame, _ = basic.load_frame(vid_filepath, f, bokeh=False)
    # converts to HSV format
    val = basic.get_val_channel(frame)
    # highlights bubbles and shows each step in the process (6 total)
    all_steps = highlight_method(val, *params, ret_all_steps=True)
    im_diff, thresh_bw, closed_bw, bubble_bw, bubble = all_steps

    # collects images to display
    im_list = [bokehfy(val), bokehfy(basic.adjust_brightness(im_diff, 3.0)),
               bokehfy(thresh_bw), bokehfy(closed_bw),
              bokehfy(bubble_bw), bokehfy(bubble)]
    title_list = ['Frame {0:d}: Value Channel (HSV)'.format(f),
                  'Subtracted Reference (Value)', 'Thresholded',
                  'Binary Closing', 'Small Obj Removed', 'Holes Filled' + tag]
    p_grid = linked_frames(im_list, pix_per_um, fig_size_red, shape=(2,3),
                             brightness=3.0, title_list=title_list)

    return p_grid
