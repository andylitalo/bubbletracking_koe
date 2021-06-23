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
from bokeh.models import ColorBar, LinearColorMapper

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


def format_frame(frame, pix_per_um, fig_size_red, brightness=1.0, title=None, backend='webgl'):
    """LEGACY--format_im() is preferred"""
    frame = basic.adjust_brightness(frame, brightness)
    width = frame.shape[1]
    height= frame.shape[0]
    width_um = int(width / pix_per_um)
    height_um = int(height / pix_per_um)
    width_fig = int(width*fig_size_red)
    height_fig = int(height*fig_size_red)
    p = figure(x_range=(0,width_um), y_range=(0,height_um), output_backend=backend,
               width=width_fig, height=height_fig, title=title)
    p.xaxis.axis_label = 'width [um]'
    p.yaxis.axis_label = 'height [um]'
    im = p.image_rgba(image=[frame], x=0, y=0, dw=width_um, dh=height_um)

    return p, im


def format_im(im, pix_per_um, x_range=None, y_range=None, scale_fig=1, 
              scale_height=1.4, brightness=1, palette='Turbo256', 
              cmap_range='from zero', hide_colorbar=False, show_fig=True,
             title='', t_fs=18, ax_fs=16, tk_fs=12, cb_fs=14):
    """
    Formats image for Bokeh plot. Allows user to view image with appropriate
    size and aspect ratio.
    
    Pass numpy array as image--*not bokehfy'd*.
    
    cmap_range : string, opt (default='from zero')
        Name of type of range for the color mapper. 
        'from zero': colors range from 0 to maximum value
        'full': colors range from minimum to maximum value
        'symmetric': colors range from -(max difference) to +(max difference),
            where max difference = maximum absolute value
    """
    # adjusts brightness for better viewing of image
    im = basic.adjust_brightness(im, brightness)
    # calculates dimensions of image and figure
    width = im.shape[1]
    height= im.shape[0]
    width_um = int(width / pix_per_um)
    height_um = int(height / pix_per_um)
    if x_range is None:
        x_range = (0, width_um)
    if y_range is None:
        y_range = (0, height_um)
    
    # creates figure
    p = figure(x_range=x_range, y_range=y_range, title=title,
               width=int(scale_fig*(x_range[1] - x_range[0])), 
               height=int(scale_fig*scale_height*(y_range[1] - y_range[0])))
    
    # creates color map if requested
    if palette is not None:
        if cmap_range == 'from zero':
            high = np.max(im)
            low = 0
        elif cmap_range == 'full':
            high = np.max(im)
            low = np.min(im)
        elif cmap_range == 'symmetric':
            max_diff = np.max(np.abs(im))
            high = max_diff
            low = -max_diff
        else:
            print('cmap_range not recognized as part of accepted list {''from zero'', ''full'', ''symmetric''}')
        # creates color mapper
        color_mapper = LinearColorMapper(palette=palette, low=low, high=high)
    else:
        color_mapper = None

    # shows image on plot
    if len(im.shape) == 2:
        p.image(image=[im],             # image data
                 x=0,               # lower left x coord
                 y=0,               # lower left y coord
                 dw=width_um,        # *data space* width of image
                 dh=height_um,        # *data space* height of image
        )
    elif len(im.shape) == 3:
        im_rgba = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
        p.image_rgba(image=[im_rgba],             # image data
                 x=0,               # lower left x coord
                 y=0,               # lower left y coord
                 dw=width_um,        # *data space* width of image
                 dh=height_um,        # *data space* height of image
        )
    elif len(im.shape) == 4:
        p.image_rgba(image=[im],             # image data
                 x=0,               # lower left x coord
                 y=0,               # lower left y coord
                 dw=width_um,        # *data space* width of image
                 dh=height_um,        # *data space* height of image
        )
    else:
        print('Image given to pltim.format_im() must be 2, 3, or 4 dims.')

        
    # adds colorbar to figure
    if color_mapper is not None and not hide_colorbar:
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=10)
        color_bar.major_label_text_font_size = '{0:d}pt'.format(cb_fs)
        p.add_layout(color_bar, 'left')
    
    # labels axes
    p.xaxis.axis_label = 'width [um]'
    p.yaxis.axis_label = 'height [um]'
    
    # adjusts font sizes
    p.title.text_font_size = '{0:d}pt'.format(t_fs)
    p.xaxis.axis_label_text_font_size = '{0:d}pt'.format(ax_fs)
    p.xaxis.major_label_text_font_size = '{0:d}pt'.format(tk_fs)
    p.yaxis.axis_label_text_font_size = '{0:d}pt'.format(ax_fs)
    p.yaxis.major_label_text_font_size = '{0:d}pt'.format(tk_fs)

    if show_fig:
        show(p)
        
    return p


def linked_frames(frame_list, pix_per_um, fig_size_red, shape=(2,2),
                  show_fig=True, brightness=1.0, title_list=[]):
    """
    Shows multiple frames with linked panning and zooming.
    
    Uses format_frame() (LEGACY)
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


def linked_ims(im_list, pix_per_um, shape=(2,2),
                  x_range=None, y_range=None, scale_fig=1, scale_height=1.4, 
                  brightness=1, palette='Turbo256', cmap_range='from zero', 
                  show_fig=True, title_list=[], t_fs=24, ax_fs=16, tk_fs=12, cb_fs=14):
    """
    Shows multiple frames with linked panning and zooming.
    
    Uses format_im().
    """
    # list of figures
    p = []
    # creates images
    for i, im in enumerate(im_list):
        if len(title_list) == len(im_list):
            title = title_list[i]
        p_new = format_im(im, pix_per_um, x_range=x_range, y_range=y_range, 
                             scale_fig=scale_fig, scale_height=scale_height, title=title,
                             brightness=brightness, palette=palette, cmap_range=cmap_range,
                             show_fig=False, t_fs=t_fs, ax_fs=ax_fs, tk_fs=tk_fs, cb_fs=cb_fs)
        p += [p_new]
        
    # makes grid plot
    p_grid = make_gridplot(p, shape)
    
    # shows figure
    if show_fig:
        show(p_grid)

    return p_grid


def make_gridplot(p, shape):
    """Makes gridplot of in given shape out of list of Bokeh plot objects."""
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
    
    return p_grid
