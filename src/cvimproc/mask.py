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



def create_polygon_mask(image,points):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the polygon is masked.
    """
    # Calculate the number of points needed perimeter of the polygon in
    # pixels (4 points per unit pixel)
    points = np.array(points,dtype=int)
    perimeter = cv2.arcLength(points,closed=True)
    nPoints = int(2*perimeter)
    # Generate x and y values of polygon
    x = points[:,0]; y = points[:,1]
    x,y = geo.generate_polygon(x,y,nPoints)
    points = [(int(x[i]),int(y[i])) for i in range(nPoints)]
    points = np.asarray(list(pd.unique(points)))
    mask = get_mask(x,y,image.shape)

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


def create_rect_mask_data(im, vertices, mask_file):
    """
    create mask for an image and save as pickle file
    """
    xMin = vertices[0][0]
    xMax = vertices[1][0]
    yMin = vertices[0][1]
    yMax = vertices[2][1]
    xyMinMax = np.array([xMin, xMax, yMin, yMax])
    # create mask from vertices
    mask, maskPts = create_polygon_mask(im, vertices)
    print(mask)
    # store mask data
    mask_data = {}
    mask_data['mask'] = mask
    mask_data['xyMinMax'] = xyMinMax
    # save new mask
    with open(maskFile,'wb') as f:
        pkl.dump(mask_data, f)

    return mask_data


def get_bbox(mask_data):
    """
    Returns the bounding box (max and min rows and columns) of a mask.

    Parameters
    ----------
    mask_data : dictionary
        Must at minimum contain entry 'boundary' containing 2-tuples of ints
        defining the (x,y) coordinates of the points along the boundary of the
        mask.

    Returns
    -------
    bbox : 4-tuple of ints
        (row_min, col_min, row_max, col_max)
    """
    # collects list of all rows and columns of points along boundary of mask
    rows = [pt[1] for pt in mask_data['boundary']]
    cols = [pt[0] for pt in mask_data['boundary']]
    # computes bounding box
    bbox =  (np.min(rows), np.min(cols), np.max(rows), np.max(cols))

    return bbox


def get_mask(X,Y,imageShape):
    """
    Converts arrays of x- and y-values into a mask. The x and y values must be
    made up of adjacent pixel locations to get a filled mask.
    """
    # Take only the first two dimensions of the image shape
    if len(imageShape) == 3:
        imageShape = imageShape[0:2]
    # Convert to unsigned integer type to save memory and avoid fractional
    # pixel assignment
    X = X.astype('uint16')
    Y = Y.astype('uint16')

    #Initialize mask as matrix of zeros
    mask = np.zeros(imageShape,dtype='uint8')
    # Set boundary provided by x,y values to 255 (white)
    mask[Y,X] = 255
    # Fill in the boundary (output is a boolean array)
    mask = basic.fill_holes(mask)

    return mask


def mask_image(image, mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # Apply mask depending on dimensions of image
    temp = np.shape(image)
    maskedImage = np.zeros_like(image)
    if len(temp) == 3:
        for i in range(3):
            maskedImage[:,:,i] = mask*image[:,:,i]
    else:
        maskedImage = image*mask

    return maskedImage
