# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015
@author: John
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import ctypes
import cv2

from tkinter import messagebox

# imports custom libraries
import genl.pltg as pltg
import genl.geo as geo
import cvimproc.mask as mask
import cvimproc.improc as improc



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
    dx = verts[1][0] - verts[0][0]
    dy = verts[1][1] - verts[0][1]
    # normalizes flow direction
    d = np.sqrt(dx**2 + dy**2)
    # returns flow direction in (x, y) coordinates
    flow_dir = np.array([dx / d, dy / d])

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
    im_p = improc.prep_for_mpl(im)
    # collects 2 pts from clicks defining the flow direction along inner wall
    if msg == '':
        msg = 'right-click 2 pts left-to-right along inner wall, then left-click'
    xy_vals = define_outer_edge(im_p, 'polygon', message=msg)
    # computes coordinates of vector from clicks
    dx = xy_vals[1][0]-xy_vals[0][0]
    dy = xy_vals[1][1]-xy_vals[0][1]
    # normalizes flow direction
    d = np.sqrt(dx**2 + dy**2)
    flow_dir = np.array([dx / d, dy / d])

    if return_origin:
        origin = xy_vals[0]
        return flow_dir, origin
    else:
        return flow_dir


def click_for_length(im, just_height=False, msg='', return_origin=False):
    """
    """
    # formats image for use in matplotlib's imshow
    im_p = improc.prep_for_mpl(im)
    # collects 2 pts from clicks defining the flow direction along inner wall
    if msg == '':
        msg = 'right-click 2 pts spanning desired length, then left-click'
    xy_vals = define_outer_edge(im_p, 'polygon', message=msg)
    # computes coordinates of vector from clicks
    dx = xy_vals[1][0]-xy_vals[0][0]
    dy = xy_vals[1][1]-xy_vals[0][1]
    if just_height:
        d = dy
    else:
        # normalizes flow direction
        d = np.sqrt(dx**2 + dy**2)

    return d


def click_for_next(ax, msg='Center-click for next image.'):
    """
    Allows code to continue only after user clicks.
    """
    ax.set_title(msg)
    # loops through clicks until center-click
    while True:
        pp = get_pts(1)
        if len(pp) < 1:
            break


def click_sheath_flow(im, mask_path, check=False):
    """
    User clicks mask for flow and mask for inner stream. The method then uses
    these to determine the flow axis and parameters for estimating velocity
    field.
    Parameters
    ----------
    im : (M x N x 3) or (M x N) numpy array of floats or uint8s
        Image that will be shown for the user to click the flow direction
    mask_path : string
        Filepath to destination of desired mask file.
    check : bool, optional
        If True and a file exists under mask_path, asks user to confirm
        quality of existing mask before proceeding. Otherwise, existing mask is
        always used. Default False.
    Returns
    -------
    flow_dir : 2-tuple of floats
        Unit vector of direction of flow. (row, col) if rc is True, o/w (x,y).
    mask_data : dictionary
        Contains data (mask and vertices) for full mask and inner stream mask
    """
    # asks user to click outer stream; ignores flow direction in case inner
    # walls of capillary are not visible in the image
    _, mask_data = click_flow(im, mask_path, save=False, check=check)
    # asks user to click inner stream
    flow_dir_xy, inner_mask_data = click_flow(im, mask_path,
                                region='inner stream', save=False, check=check)
    # flips flow direction to be row col
    flow_dir = flow_dir_xy[::-1]
    # extracts vertices of inner stream for use in defining inner stream width
    verts_inner = inner_mask_data['vertices']
    # stores inner stream properties in mask data dictionary
    mask_data['flow_dir'] = flow_dir
    mask_data['verts_inner'] = verts_inner
    # saves mask data
    with open(mask_path, 'wb') as f:
        pkl.dump(mask_data, f)

    return flow_dir, mask_data


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


def define_outer_edge(image,shapeType,message=''):
    """
    Displays image for user to outline the edge of a shape. Tracks clicks as
    points on the image. If shapeType is "polygon", the points will be
    connected by lines to show the polygon shape. If the shapeType is "circle",
    the points will be fit to a circle after 4 points have been selected,
    after which point the user can continue to click more points to improve
    the fit.
    Possible "shapeTypes":
    'polygon': Returns array of tuples of xy-values of vertices
    'circle': Returns radius and center of circle
    'ellipse': Returns radius1, radius2, center, and angle of rotation
    'rectangle': Returns array of tuples of xy-values of 4 vertices given two
                opposite corners
    """
    # parse input for circle
    if shapeType == 'circle':
        guess = np.shape(image)
        guess = (guess[1]/2,guess[0]/2)
    # define dictionary of shapes --> shape adjectives
    shapeAdjDict = {'circle':'Circular','ellipse':'Ellipsular',
    'polygon':'Polygonal','rectangle':'Rectangular'}
    if shapeType in shapeAdjDict:
        shapeAdj = shapeAdjDict[shapeType]
    else:
        print("Please enter a valid shape type (\"circle\",\"ellipse\"" + \
        "\"polygon\", or \"rectangle\").")
        return

    # Initialize point lists and show image
    x = []; y = []
    figName = 'Define %s Edge - Center click when satisfied' %shapeAdj
    plt.figure(figName)
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
        plt.plot(x,y,'r.',alpha=0.5)
        plt.draw()
        # Perform fitting and drawing of fitted shape
        if shapeType == 'circle':
            if len(x) > 2:
                xp = np.array(x)
                yp = np.array(y)
                R,center,temp =  geo.fit_circle(xp,yp,guess)
                guess = center
                X,Y = geo.generate_circle(R,center)
                plt.plot(X,Y,'y-',alpha=0.5)
                plt.plot(center[0],center[1],'yx',alpha=0.5)
                plt.draw()
        elif shapeType == 'ellipse':
            if len(x) > 3:
                xp = np.array(x)
                yp = np.array(y)
                R1,R2,center,theta =  geo.fit_ellipse(xp,yp)
                X,Y = geo.generate_ellipse(R1,R2,center,theta)
                plt.plot(X,Y,'y-',alpha=0.5)
                plt.plot(center[0],center[1],'yx',alpha=0.5)
                plt.draw()
        elif shapeType == 'polygon':
            plt.plot(x,y,'y-',alpha=0.5)
            plt.draw()
        elif shapeType == 'rectangle':
            # need 2 points to define rectangle
            if len(x) == 2:
                # generate points defining rectangle containing xp,yp as opposite vertices
                X,Y = geo.generate_rectangle(x, y)
                # plot on figure
                plt.plot(X,Y,'y-', alpha=0.5)
                plt.draw()


    plt.close()
    if shapeType == "circle":
        return R,center
    elif shapeType == 'ellipse':
        return R1,R2,center,theta
    elif shapeType == "polygon":
        xyVals = [(x[i],y[i]) for i in range(len(x))]
        return xyVals
    elif shapeType == 'rectangle':
        # returns (x,y) values of 4 vertices starting with upper left in clockwise order
        xyVals = [(np.min(X),np.min(Y)), (np.max(X),np.min(Y)),
                  (np.max(X),np.max(Y)), (np.min(X),np.max(Y))]
        return xyVals


def get_rect_mask_data(im, maskFile, check=False, yes=6,
                        msg="Click opposing corners of rectangle ' + \
                        'outlining desired region."):
    """
    Shows user masks overlayed on given image and asks through a dialog box
    if they are acceptable. Returns True for 'yes' and False for 'no'.
    """
    try:
        with open(maskFile, 'rb') as f:
            maskData = pkl.load(f)
    except:
        print('Mask file not found, please create it now.')
        # asks user to click vertices
        vertices = define_outer_edge(im, 'rectangle', message=msg)
        maskData = mask.create_rect_mask_data(im, vertices, mask_file)

    while check:
        plt.figure('Evaluate accuracy of predrawn masks for your video')
        maskedImage = mask.mask_image(im,maskData['mask'])
        temp = np.dstack((maskedImage,im,im))
        plt.imshow(temp)

        response = ctypes.windll.user32.MessageBoxA(0, 'Do you wish to keep' + \
                            ' the current mask?','User Input Required', 4)

        if response == yes: # 6 means yes
            plt.close()
            return maskData

        else: # 7 means no
            print('Existing mask rejected, please create new one now.')
            msg = "Click opposing corners of rectangle outlining desired region."
            # asks user to click vertices
            vertices = define_outer_edge(im, 'rectangle', message=msg)
            maskData = mask.create_rect_mask_data(im, vertices, mask_file)

    plt.close()

    return maskData


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
    image = basic.extract_frame(v,1,hMatrix=hMatrix)
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
        plt.plot(center[0],center[1],'bx')
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


def get_pix_per_um(im, l_um):
    """
    Given an image, the user clicks points defining a line segment of a known
    length. The length of that line segment is calculated and an approximate
    conversion of pixels to actual distance is obtained.
    Parameters:
        im : 2D or 3D array
            Image with a known distance to be measured in pixels
        l_um : float
            Length of a known distance in the given image [um]
    Returns:
        pix_per_um : float
            Number of pixels per um in the image.
    """
    # Format window for image
    fig_name = 'Click line segment across inner diameter of capillary'
    msg = 'Right-click the endpoints of a line segment perpendicular ' +\
    'to the capillary spanning its inner diameter.'
    plt.figure(fig_name)
    plt.rcParams.update({'figure.autolayout': True})
    plt.set_cmap('gray')
    plt.imshow(im)
    plt.axis('image')
    plt.axis('off')
    plt.title(msg)
    # Initialize list to store clicked points
    pts = []
    while True:
        clicked_pts = get_pts(num_pts=1, im=im)
        lims = plt.axis()
        if len(clicked_pts) < 1:
            break
        else:
            # add points to list
            pts += clicked_pts[0]
        # Reset the plot
        plt.cla()
        plot.no_ticks(im)
        plt.title(msg)
        plt.axis(lims)
        # Plot new points
        plt.plot(clicked_pts[0], clicked_pts[1], 'r.', alpha=0.5)
        # Perform fitting and drawing of fitted shape
        if len(pts) == 2:
            xp = np.array([pt[0] for pt in pts])
            yp = np.array([pt[1] for pt in pts])
            plt.plot(xp, yp,'y-',alpha=0.5)
    l_pix = np.linalg.norm(pts[1]-pts[0])
    pix_per_um = l_pix / l_um
    # close figure
    plt.close()

    return pix_per_um

if __name__ == '__main__':
    pass
