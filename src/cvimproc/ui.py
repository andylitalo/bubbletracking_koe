# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015
@author: John
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import ctypes

from tkinter import messagebox

# imports custom libraries
import genl.pltg as pltg
import genl.geo as geo
import cvimproc.mask as mask
import cvimproc.basic as basic



def click_flow(im, mask_path, region='ROI', save=True, check=False):
    """
    Same as click_flow_dir but requests two additional clicks along the other
    inner wall of the capillary (or interface of the inner stream if both inner
    walls are not visible) to define the mask for the flow.
    If using an IDE like Spyder or Jupyter Notebook, user must enter the inline
    command "%matplotlib qt" to display a clickable window for this function.
    Parameters
    ----------
    im : (M x N x 3) or (M x N) numpy array of floats or uint8s
        Image that will be shown for the user to click the flow direction
    mask_path : string
        Filepath to destination of desired mask file.
    region : string, optional
        Region of image that the user should click. Default 'ROI' for "region
        of interest," i.e., the region that will be considered in analysis.
    save : bool, optional
        If True, will save mask data at mask_path. Default True.
    check : bool, optional
        If True and a file exists under mask_path, asks user to confirm
        quality of existing mask before proceeding. Otherwise, existing mask is
        always used. Default False.
    Returns
    -------
    flow_dir : 2-tuple of floats
        Unit vector in (x,y) indicating direction of flow
    mask_data : dictionary
        Contains coordinates of individual points (x,y) defining the vertices of
        the mask and the (M x N) array of the mask.
    """
    mask_data = get_polygonal_mask_data(im, mask_path, check=check,
                                                save=save)
    verts = mask_data['vertices']
    # computes coordinates of flow vector from first two clicks
    dx = verts[1][1] - verts[0][1]
    dy = verts[1][0] - verts[0][0]
    # normalizes flow direction
    d = np.sqrt(dx**2 + dy**2)
    # returns flow direction in (row, column) coordinates
    flow_dir = np.array([dy / d, dx / d])

    return flow_dir, mask_data


def click_flow_dir(im, msg='', return_origin=False):
    """
    User clicks along the inner wall of the capillary to indicate the flow
    direction in a vector (tuple). User must enter the inline command
    "%matplotlib qt" to get a pop-out plot for using this function.
    Parameters
    ----------
    im : (M x N x 3) or (M x N) numpy array of floats or uint8s
        Image that will be shown for the user to click the flow direction
    return_origin : bool, optional
        If True, method also returns coordinates of first click.
    Returns
    -------
    flow_dir : 2-tuple of floats
        Unit vector in (x,y) indicating direction of flow
    origin : 2-tuple of floats, optional
        Returned if return_origin is True. Gives (x,y) coords. of first click.
    """
    # formats image for use in matplotlib's imshow
    im_p = basic.prep_for_mpl(im)
    # collects 2 pts from clicks defining the flow direction along inner wall
    if msg == '':
        msg = 'right-click 2 pts left-to-right along inner wall, then left-click'
    pts = define_outer_edge(im_p, 'polygon', message=msg)
    # computes coordinates of vector from clicks; points are (row, column)
    dx = pts[1][1] - pts[0][1]
    dy = pts[1][0] - pts[0][0]
    # normalizes flow direction
    d = np.sqrt(dx**2 + dy**2)
    # gives flow direction in (row, column) format
    flow_dir = np.array([dy / d, dx / d])

    if return_origin:
        origin = pts[0]
        return flow_dir, origin
    else:
        return flow_dir


def click_for_length(im, just_height=False, msg='', return_origin=False):
    """
    """
    # formats image for use in matplotlib's imshow
    im_p = basic.prep_for_mpl(im)
    # collects 2 pts from clicks defining the flow direction along inner wall
    if msg == '':
        msg = 'right-click 2 pts spanning desired length, then left-click'
    pt1, pt2 = define_outer_edge(im_p, 'polygon', message=msg)
    # computes coordinates of vector from clicks; points are (row, column)
    dx = pt2[1] - pt1[1]
    dy = pt2[0] - pt1[0]
    if just_height:
        d = dy
    else:
        # normalizes flow direction
        d = np.sqrt(dx**2 + dy**2)

    return d


def click_z_origin(im):
    """
    Records user clicks that indicate the desired origin and z-axis in an
    image. These can be used to calculate the r-coordinate measured off the
    z-axis.
    Parameters
    ----------
    im : (M x N x 3) or (M x N) numpy array of floats or uint8s
        Image that will be shown for the user to click origin and z axis.
    Returns
    -------
    z : 2-tuple of float
        Normalized vector indicating direction of z-axis (flow axis)
    o : 2-tuple of float
        (x,y) coordinates of origin of central axis of capillary
    """
    # creates messages for first and second rounds of clicking
    msg_lo = 'right-click 2 pts left-to-right along lower inner wall, then left-click'
    msg_hi = 'right-click 2 pts left-to-right along upper inner wall, then left-click'
    # collects z-axis direction and origin from clicks along lower and upper walls
    z_lo, o_lo = click_flow_dir(im, msg=msg_lo, return_origin=True)
    z_hi, o_hi = click_flow_dir(im, msg=msg_hi, return_origin=True)
    # averages z-axis for better estimate of true value
    z_raw = ((z_lo[0]+z_hi[0])/2, (z_lo[1]+z_hi[1])/2)
    # normalizes z-axis
    dz = np.sqrt(z_raw[0]**2 + z_raw[1]**2)
    z = z_raw / dz
    # rows of lower and upper inner walls
    row_lo = o_lo[1]
    row_hi = o_hi[1]

    return z, row_lo, row_hi


def define_outer_edge(image, shape_type, message=''):
    """
    Displays image for user to outline the edge of a shape. Tracks clicks as
    points on the image. If shape type is "polygon", the points will be
    connected by lines to show the polygon shape. If the shape type is "circle",
    the points will be fit to a circle after 4 points have been selected,
    after which point the user can continue to click more points to improve
    the fit.

    Uses x and y coordinates when plotting.

    Parameters
    ----------
    image : (M x N) numpy array of uint8s
        Image on which to define the desired shape
    shape_type : string
        Describes the type of shape to interpret the clicked points as. 
        Options include:
            'polygon': Returns array of tuples of (row,col)-values of vertices
            'circle': Returns radius and center (row,col) of circle
            'ellipse': Returns radius1, radius2, center (row,col), and angle of rotation
            'rectangle': Returns array of tuples of (row,col)-values of 4 vertices given two
                        opposite corners
    message : string
        message to display for user while clicking

    Returns
    -------
    if shape_type == 'circle':
        R : float
            radius of circle
        center : 2-tuple of ints
            (row, column) of center of circle
    if shape_type == 'ellipse':
        R1, R2 : float
            major and minor axes of ellipse
        center : 2-tuple of ints
            (row, column) of center of ellipse
        theta : float
            angle of orientation of ellipse (radians)
    if shape_type == 'polygon':
        pts : list of 2-tuples of ints
            (row, column) of points defining boundary of polygon
    if shape_type == 'rectangle':
        vertices : list of 4 2-tuples of ints
            four vertices of rectangle, starting with upper left and 
            proceeding clockwise
    """
    # parse input for circle
    if shape_type == 'circle':
        guess = np.shape(image)
        guess = (guess[1]/2, guess[0]/2)
    # define dictionary of shapes --> shape adjectives
    shape_adjectives = {'circle' : 'Circular', 'ellipse' : 'Ellipsular',
    'polygon' : 'Polygonal', 'rectangle' : 'Rectangular'}
    if shape_type in shape_adjectives:
        shape_adjective = shape_adjectives[shape_type]
    else:
        print("Please enter a valid shape type (\"circle\",\"ellipse\"" + \
        "\"polygon\", or \"rectangle\").")
        return

    # Initialize point lists and show image
    x = []; y = []
    fig_name = 'Define {0:s} Edge - Center click when satisfied'.format(shape_adjective)
    plt.figure(fig_name)
    plt.rcParams.update({'figure.autolayout': True})
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.axis('image')
    plt.axis('off')
    plt.title(message)

    # Get data points until the user closes the figure or center-clicks
    while True:
        pp = get_pts(1)
        lims = plt.axis()
        if len(pp) < 1:
            break
        else:
            # extract tuple of (x,y) from list
            pp = pp[0]
        # Reset the plot
        plt.cla()
        pltg.no_ticks(image)
        plt.title(message)
        plt.axis(lims)
        # Add the new point to the list of points and plot them
        x += [pp[0]]; y += [pp[1]]
        plt.plot(x, y, 'r.', alpha=0.5)
        plt.draw()
        # Perform fitting and drawing of fitted shape
        if shape_type == 'circle':
            if len(x) > 2:
                xp = np.array(x)
                yp = np.array(y)
                R, center, temp =  geo.fit_circle(yp, xp, guess)
                guess = center
                rows, cols = geo.generate_circle(R, center)
                plt.plot(cols, rows, 'y-', alpha=0.5)
                plt.plot(center[1], center[0], 'yx', alpha=0.5)
                plt.draw()
        elif shape_type == 'ellipse':
            if len(x) > 3:
                xp = np.array(x)
                yp = np.array(y)
                R1, R2, center, theta =  geo.fit_ellipse(yp, xp)
                rows, cols = geo.generate_ellipse(R1, R2, center, theta)
                plt.plot(cols, rows, 'y-', alpha=0.5)
                plt.plot(center[1], center[0], 'yx', alpha=0.5)
                plt.draw()
        elif shape_type == 'polygon':
            plt.plot(x, y, 'y-', alpha=0.5)
            plt.draw()
        elif shape_type == 'rectangle':
            # need 2 points to define rectangle
            if len(x) == 2:
                # generate points defining rectangle containing xp,yp as opposite vertices
                rows, cols = geo.generate_rectangle(x, y)
                # plot on figure
                plt.plot(cols, rows, 'y-', alpha=0.5)
                plt.draw()
    plt.close()

    # collects variables to return for each case
    if shape_type == "circle":
        return R, center
    elif shape_type == 'ellipse':
        return R1, R2, center, theta
    elif shape_type == "polygon":
        pts = [(row, col) for row, col in zip(y, x)]
        return pts
    elif shape_type == 'rectangle':
        # returns (row, col)  of 4 vertices starting with upper left in clockwise order
        vertices = [(np.min(rows), np.min(cols)), (np.min(rows), np.max(cols)),
                  (np.max(rows), np.max(cols)), (np.max(rows), np.min(cols))]
        return vertices


def get_polygonal_mask_data(im, mask_file, check=False, save=True,
                                                        msg='click vertices'):
    """
    Shows user masks overlayed on given image and asks through a dialog box
    if they are acceptable. Returns True for 'yes' and False for 'no'.
    """
    try:
        with open(mask_file, 'rb') as f:
            mask_data = pkl.load(f)
    except:
        print('Mask file not found, please create it now.')
        vertices = define_outer_edge(im, 'polygon', message=msg)
        mask_data = mask.create_polygonal_mask_data(im, vertices, mask_file,
                                                                save=save)
    while check:
        plt.figure('Evaluate accuracy of predrawn masks for your video')
        plt.rcParams.update({'figure.autolayout': True})
        plt.set_cmap('gray')
        masked_image = mask.mask_image(im, mask_data['mask'])
        plt.imshow(masked_image)
        plt.axis('image')
        plt.axis('off')

        # ask if user wishes to keep current mask (header, question)
        response = messagebox.askyesno('User Input Required', 'Do you wish to keep' + \
                            ' the current mask?')
        if response:
            plt.close()
            return mask_data

        else:
            print('Existing mask rejected, please create new one now.')
            vertices = define_outer_edge(im, 'polygon', message=msg)
            mask_data = mask.create_polygonal_mask_data(im, vertices, mask_file)

    plt.close()

    return mask_data


def get_mask_data(maskFile,v,hMatrix=None,check=False):
    """
    Shows user masks overlayed on given image and asks through a dialog box
    if they are acceptable. Returns True for 'yes' and False for 'no'.
    """
    # Parse input parameters
    image = basic.extract_frame(v, 1, hMatrix=hMatrix)
    try:
        with open(maskFile) as f:
            maskData = pkl.load(f)
    except:
        print('Mask file not found, please create it now.')
        maskData = mask.create_mask_data(image,maskFile)

    while check:
        plt.figure('Evaluate accuracy of predrawn masks for your video')
        maskedImage = mask.mask_image(image,maskData['mask'])
        temp = np.dstack((maskedImage,image,image))
        plt.imshow(temp)
        center = maskData['diskCenter']
        plt.plot(center[1], center[0], 'bx')
        plt.axis('image')

        response = ctypes.windll.user32.MessageBoxA(0, 'Do you wish to keep' + \
                            ' the current mask?','User Input Required', 4)
        plt.close()
        if response == 6: # 6 means yes
            return maskData

        else: # 7 means no
            print('Existing mask rejected, please create new one now.')
            maskData = mask.create_mask_data(image,maskFile)

    return maskData


def get_pts(num_pts=1,im=None):
    """
    Alter the built in ginput function in matplotlib.pyplot for custom use.
    This version switches the function of the left and right mouse buttons so
    that the user can pan/zoom without adding points.
    NOTE: the left mouse button still removes existing points.
    Parameters:
        num_pts : int, optional
            number of points to get from user clicks.
        im : 2D or 3D array, optional
            If image is given, it will be shown and used for clicking.
            Otherwise, the current image will be used.
    Returns:
        pts : list of tuples
            (x,y) coordinates of clicks on image
    """
    if im is not None:
        plt.imshow(im)
        plt.axis('image')

    pts = plt.ginput(n=num_pts,mouse_add=3, mouse_pop=2, mouse_stop=1,
                    timeout=0)
    return pts