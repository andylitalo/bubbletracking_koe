# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:11:26 2020

@author: Andy
mask.py contains functions related to masks of images.
The mask_im function is also in improc.py since most functions
only require it. These functions are largely obsolte and are kept
in this document so libraries with legacy code can still run.

"""

import numpy as np
import pandas as pd
import cv2
import pickle as pkl


# imports custom libraries
import genl.geo as geo
import cvimproc.basic as basic



def create_polygon_mask(image, points):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the polygon is masked.
    """
    # Calculates the number of points needed perimeter of the polygon in
    # pixels (4 points per unit pixel)
    points = np.array(points, dtype=int)
    perimeter = cv2.arcLength(points, closed=True)
    n_points = int(2*perimeter)
    # Generate x and y values of polygon
    rows = points[:,0]; cols = points[:,1]
    rows, cols = geo.generate_polygon(rows, cols, n_points)
    points = [(int(rows[i]), int(cols[i])) for i in range(n_points)]
    points = np.asarray(list(pd.unique(points)))
    mask = get_mask(rows, cols, image.shape)

    return mask, points


def create_polygonal_mask_data(im, vertices, mask_file, save=True):
    """
    create mask for an image and save as pickle file
    """
    # creates mask and points along boundary
    mask, boundary = create_polygon_mask(im, vertices)
    # store mask data
    mask_data = {}
    mask_data['mask'] = mask
    mask_data['boundary'] = boundary
    mask_data['vertices'] = vertices

    # save new mask
    if save:
        with open(mask_file,'wb') as f:
            pkl.dump(mask_data, f)

    return mask_data


def get_bbox(mask_data):
    """
    Returns the bounding box (max and min rows and columns) of a mask.

    Parameters
    ----------
    mask_data : dictionary
        Must at minimum contain entry 'boundary' containing 2-tuples of ints
        defining the (row, col) coordinates of the points along the boundary of the
        mask.

    Returns
    -------
    bbox : 4-tuple of ints
        (row_min, col_min, row_max, col_max)
    """
    # collects list of all rows and columns of points along boundary of mask
    rows = [pt[0] for pt in mask_data['boundary']]
    cols = [pt[1] for pt in mask_data['boundary']]
    # computes bounding box
    bbox =  (np.min(rows), np.min(cols), np.max(rows), np.max(cols))

    return bbox


def get_mask(rows, cols, im_shape):
    """
    Converts arrays of x- and y-values into a mask. The x and y values must be
    made up of adjacent pixel locations to get a filled mask.
    """
    # Take only the first two dimensions of the image shape
    if len(im_shape) == 3:
        im_shape = im_shape[:2]
    # Convert to unsigned integer type to save memory and avoid fractional
    # pixel assignment
    rows = rows.astype('uint16')
    cols = cols.astype('uint16')

    #Initialize mask as matrix of zeros
    mask = np.zeros(im_shape, dtype='uint8')
    # Set boundary provided by row, col values to 255 (white)
    mask[rows, cols] = 255
    # Fill in the boundary (output is a boolean array)
    mask = basic.fill_holes(mask)

    return mask


def mask_image(image, mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # ensures that each color channel of the image has the same dimensions as mask
    assert image.shape[:2] == mask.shape, 'image and mask must have same shape'
    # Applies mask depending on dimensions of image
    im_masked = np.zeros_like(image)
    if len(image.shape) == 3:
        for i in range(3):
            im_masked[:,:,i] = np.where(mask, image[:,:,i], 0)
    else:
        im_masked = np.where(mask, image, 0)

    return im_masked
