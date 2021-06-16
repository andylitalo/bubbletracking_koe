"""
improc.py contains definitions of methods for image-processing
using OpenCV EXCLUSIVELY.
"""

# imports standard libraries
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
from collections import OrderedDict
import time

# imports image-processing-specific libraries
import cv2

# imports custom image processing libraries
import cvimproc.mask as mask
import cvimproc.basic as basic

# imports general custom libraries
import genl.fn as fn
import genl.geo as geo
from genl.conversions import *
import cvimproc.pltim as pltim

# imports custom classes (with reload clauses)
from classes.classes import TrackedObject, FileVideoStream

# import custom opencv video processing methods
import cvvidproc

############################# METHOD DEFINITIONS ##############################


def appears_in_consec_frames(obj, n_consec=1):
    """Returns True if object appears in consecutive frames."""
    return len(np.where(np.diff(obj.get_props('frame'))==1)[0]) >= n_consec


def average_rgb(im):
    """
    Given an RGB image, averages RGB to produce b-w image.
    Note that the order of RGB is not important for this.

    Parameters
    ----------
    im : (M x N x 3) numpy array of floats or ints
        RGB image to average

    Returns
    -------
    im_rgb_avg : (M x N) numpy array of floats or ints
        Black-and-white image of averaged RGB

    """
    # ensures that the image is indeed in suitable RGB format
    assert im.shape[2] == 3, "Image must be RGB (M x N x 3 array)."
    # computes average over RGB values
    res = np.round(np.mean(im, 2))
    # converts to uint8
    return res.astype('uint8')


def assign_obj(frame_bw, f, objs_prev, objs_archive, ID_curr, fps,
                   d_fn, d_fn_kwargs, width_border=2, min_size_reg=0,
               row_lo=0, row_hi=0, remember_objs=False):
    """
    Assigns objects with unique IDs to the labeled objects on the video
    frame provided. This method is used on a single frame in the context of
    processing an entire video and uses information from the previous frame.
    Inspired by and partially copied from PyImageSearch [1].

    Only registers objects above the minimum registration size.
    Once registered, an object no longer needs to remain above the 
    area threshold.

    Updates objs_prev and objs_archive in place.

    ***OpenCV-compatible version: labels frame instead of receiving labeled
    frame by using cv2.connectedComponentsWithStats.

    Parameters
    ----------
    frame_bw : (M x N) numpy array of uint8
        Binarized video frame
    f : int
        Frame number from video
    objs_prev : Dictionary of dictionaries
        Contains dictionaries of properties of objects from the previous frame
        labeled by ID number???
    objs_archive : dictionary of TrackedObj objects
        Dictionary of objects from all previous frames
    ID_curr : int
        Next ID number to be assigned (increasing order)
    flow_dir : numpy array of 2 floats
        Unit vector indicating the flow direction. Should be in (row, col).
    fps : float
        Frames per second of video
    pix_per_um : float
        Conversion of pixels per micron (um)
    width_border : int
        Number of pixels to remove from border for image processing in
        ee()
    row_lo : int
        Row of lower inner wall--currently not implemented
    row_hi : int
        Row of upper inner wall--currently not implemented
    v_max : float
        Maximum velocity expected due to Poiseuille flow [pix/s]
    min_size_reg : int
        Objects must have a greater area than this threshold to be registered.

    Returns
    -------
    ID_curr : int
        Updated value of next ID number to assign.

    References
    ----------
    .. [1] https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
    # computes frame dimesions
    frame_dim = frame_bw.shape
    # measures region props of each object in image
    objs_curr = region_props(frame_bw, n_frame=f, width_border=width_border)

    # if no objects seen in previous frame, assigns objects in current frame
    # to new IDs
    if len(objs_prev) == 0:
        for i in range(len(objs_curr)):
            # only registers objects larger than a minimum size
            if objs_curr[i]['area'] >= min_size_reg:
                objs_prev[ID_curr] = objs_curr[i]
                ID_curr += 1

    # if no objects in current frame, removes objects from dictionary of
    # objects in the previous frame
    elif len(objs_curr) == 0:
        for ID in list(objs_prev.keys()):
            # predicts the next centroid for the object based on previous velocity
            centroid_pred = objs_archive[ID].predict_centroid(f)
            # if the most recent (possibly predicted) centroid is out of bounds,
            # or if our prediction comes from a single data point (so the
            # velocity is uncertain),
            # then the object is deleted from the dictionary
            if lost_obj(centroid_pred, frame_bw, ID, objs_archive) or (not remember_objs):
                del objs_prev[ID]
            # otherwise, predicts next centroid, keeping other props the same
            else:
                objs_prev[ID]['frame'] = f
                objs_prev[ID]['centroid'] = centroid_pred

    # otherwise, assigns objects in current frames to previous objects based
    # on provided distance function
    else:
        # grabs the set of object IDs from the previous frame
        IDs = list(objs_prev.keys())
        # computes M x N matrix of distances (M = # objects in previous frame,
        # N = # objects in current frame)
        d_mat = obj_d_mat(list(objs_prev.values()),
                             objs_curr, d_fn, d_fn_kwargs)

        ### SOURCE: Much of the next is directly copied from [1]
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = d_mat.min(axis=1).argsort()
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = d_mat.argmin(axis=1)[rows]
        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        rows_used = set()
        cols_used = set()

        # loops over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignores it
            if row in rows_used or col in cols_used:
                continue
            # also ignores pairings where the second object is upstream
            # these pairings are marked with a penalty that is
            # larger than the largest distance across the frame
            d_longest = np.linalg.norm(frame_bw.shape)
            if d_mat[row, col] > d_longest:
                continue

            # otherwise, grabs the object ID for the current row,
            # set its new centroid, and reset the disappeared
            # counter
            ID = IDs[row]
            objs_prev[ID] = objs_curr[col]
            # indicates that we have examined each of the row and
            # column indexes, respectively
            rows_used.add(row)
            cols_used.add(col)

        # computes both the row and column index we have NOT yet
        # examined
        rows_unused = set(range(0, d_mat.shape[0])).difference(rows_used)
        cols_unused = set(range(0, d_mat.shape[1])).difference(cols_used)

        # loops over the unused row indexes to remove objects that disappeared
        for row in rows_unused:
            # grabs the object ID for the corresponding row
            # index and save to archive
            ID = IDs[row]
            # predicts next centroid
            centroid_pred = objs_archive[ID].predict_centroid(f)
            # deletes object if centroid is out of bounds or if prediction is
            # based on just 1 data point (so velocity is uncertain)
            if lost_obj(centroid_pred, frame_bw, ID, objs_archive) or (not remember_objs):
                del objs_prev[ID]
            # otherwise, predicts next centroid, keeping other props the same
            else:
                objs_prev[ID]['frame'] = f
                objs_prev[ID]['centroid'] = centroid_pred


        # registers each unregistered new input centroid as a object seen
        for col in cols_unused:
            # adds only objects above threshold
            if objs_curr[col]['area'] >= min_size_reg:
                objs_prev[ID_curr] = objs_curr[col]
                ID_curr += 1

    # archives objects from this frame in order of increasing ID
    for ID in objs_prev.keys():
        # creates new ordered dictionary of objects if new object
        if ID == len(objs_archive):
            metadata = {'ID' : ID, 'fps' : fps, 'frame_dim' : frame_dim}
            objs_archive[ID] = TrackedObject(metadata, props_raw=objs_prev[ID])
        elif ID < len(objs_archive):
            objs_archive[ID].add_props(objs_prev[ID])
        else:
            print('In assign_obj(), IDs looped out of order while saving to archive.')

    return ID_curr


def compute_bkgd_mean(vid_path, num_frames=100, print_freq=10):
    """
    Same as compute_bkgd_med() but computes mean instead of median. Very slow.
    """
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    if not ret:
        return None
    # computes the mean
    total = np.zeros(frame.shape)
    n = 0
    while ret:
        total += frame.astype('int')
        n += 1
        if (n % print_freq) == 0:
            print('{0:d} frames complete for compute_bkgd_mean()'.format(n))
        if n == num_frames:
            break
        # reads next frame
        ret, frame = cap.read()
    # computes the mean
    bkgd_mean = total / n
    # formats result as an image (unsigned 8-bit integer)
    bkgd_mean = bkgd_mean.astype('uint8')
    print('Computed mean of {0:s}'.format(vid_path))

    # takes value channel if color image provided
    if len(bkgd_mean.shape) == 3:
        bkgd_mean = basic.get_val_channel(bkgd_mean)

    return bkgd_mean

    
def compute_bkgd_med(vid_path, num_frames=100):
    """
    Same as compute_bkgd_med_thread() but does not use threading. More reliable
    and predictable, but slower.
    """
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    if not ret:
        return None
    # computes the median
    frame_trio = proc_frames(cap, med_alg, ([frame],),
                                    num_frames=num_frames)
    print('finished')
    # the median value of pixels is the first object in the frame trio list
    bkgd_med = frame_trio[0][0].astype('uint8')

    # takes value channel if color image provided
    if len(bkgd_med.shape) == 3:
        bkgd_med = basic.get_val_channel(bkgd_med)

    return bkgd_med

def compute_bkgd_med_thread(vid_path, vid_is_grayscale, num_frames=100, 
			crop_x=0, crop_y=0, crop_width=0, crop_height=0,
			max_threads=-1):
    """
    Calls multithreaded bkgd algo.
    """

    # computes the median
    vidpack = cvvidproc.VidBgPack(
        vid_path = vid_path,
        bg_algo = 'hist',
        max_threads = max_threads, #(default = -1)
        frame_limit = num_frames, #(default = -1 -> all frames),
        grayscale = True,
        vid_is_grayscale = vid_is_grayscale,
        crop_x = crop_x,
        crop_y = crop_y,
        crop_width = crop_width, #(default = 0)
        crop_height = crop_height, #(default = 0)
        token_storage_limit = 200,
        print_timing_report = True)

    print('getting video background')
    start_time = time.time()

    bkgd_med = cvvidproc.GetVideoBackground(vidpack)

    end_time = time.time()
    print('video background obtained ({0:f} s)'.format(end_time - start_time))

    # takes value channel if color image provided
    if len(bkgd_med.shape) == 3:
        bkgd_med = get_val_channel(bkgd_med)

    return bkgd_med


def d_euclid_bw_obj(obj1, obj2):
    """
    Computes the distance between each pair of points in the two sets
    perpendicular to the axis using the Euclidean distance (L2-norm).
    
    Parameters
    ----------
    obj1, obj2 : dictionaryies
        Dictionaries of object properties containing 'centroid'
        
    Returns
    -------
    d : float
        Distance between the centroids of object 1 and object 2
    """
    # extracts centroids of the two objects in (row, col) format [pixels]
    c1 = np.array(obj1['centroid'])
    c2 = np.array(obj2['centroid'])
    
    # computes Euclidean distance (L2-norm) between centroids
    # using % timeit, this is 3x faster than np.linalg.norm()
    d = np.sqrt( (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 )
    
    return d



def find_label(frame_labeled, rc, cc):
    """
    Returns the label for the object with the given centroid coordinates.
    Typically, this is just the value of the labeled frame at the integer
    values of the centroid coordinates, but for concave object shapes, the
    integer values of the centroid coordinates might not designate a pixel
    that is inside the object. This algortihm continues the search to find
    a pixel that is indeed inside the object and get its label.

    Parameters
    ----------
    frame_labeled : (M x N) numpy array of uint8
        Image of objects labeled by 1-indexed integers (background is 0)
    rc : float
        Row of the centroid of the object whose label is desired.
    cc : float
        Column of centroid of the object label is desiredRIPTION.

    Returns
    -------
    label : int
        Label of the image with centroid at (rc, cc). -1 indicates failure.

    """
    # extracts number and rows and columns of frame
    num_rows, num_cols = frame_labeled.shape
    # lists steps to take from centroid to find pixel in the labeled object
    steps = [(0,0), (0,1), (1,0), (1,1)]
    # searches around centroid for a labeled point
    for step in steps:
        label = frame_labeled[min(int(rc)+step[0], num_rows-1),
                                min(int(cc)+step[1], num_cols-1)]
        if label != 0:
            return label

    # failed to find non-zero label--returns -1 to indicate failure
    return -1


def frame_and_fill(im, w=2):
    """
    Frames image with border to fill in holes cut off at the edge. Without
    adding a frame, a hole in an object that is along the edge will not be
    viewed as a hole to be filled by fill_holes
    since a "hole" must be completely enclosed by white object pixels.

    Parameters
    ----------
    im : numpy array of uint8
        Image whose holes are to be filled. 0s and 255s
    w : int, opt (default=2)
        Width of border used to frame image to fill in holes cut off at edge.

    Returns
    -------
    im : numpy array of uint8
        Image with holes filled, including those cut off at border. 0s and 255s

    """
        # removes any white pixels slightly inside the border of the image
    mask_full = np.ones([im.shape[0]-2*w, im.shape[1]-2*w])
    mask_deframe = np.zeros_like(im)
    mask_deframe[w:-w,w:-w] = mask_full
    mask_frame_sides = np.logical_not(mask_deframe)
    # make the tops and bottoms black so only the sides are kept
    mask_frame_sides[0:w,:] = 0
    mask_frame_sides[-w:-1,:] = 0
    # frames sides of filled object image
    im_framed = np.logical_or(im, mask_frame_sides)
    # fills in open space in the middle of the object that might not get
    # filled if object is on the edge of the frame (because then that
    # open space is not completely bounded)
    im_filled = basic.fill_holes(im_framed)
    im = mask_im(im_filled, np.logical_not(mask_frame_sides))

    return im


def get_angle_correction(im_labeled):
    """
    correction of length due to offset angle of stream
    """
    rows, cols = np.where(im_labeled)
    # upper left (row, col)
    ul = (np.min(rows[cols==np.min(cols)]), np.min(cols))
    # upper right (row, col)
    ur = (np.min(rows[cols==np.max(cols)]), np.max(cols))
    # bottom left (row, col)
    bl = (np.max(rows[cols==np.min(cols)]), np.min(cols))
    # bottom right (row, col)
    br = (np.max(rows[cols==np.max(cols)]), np.max(cols))
    # angle along upper part of stream
    th_u = np.arctan((ul[0]-ur[0])/(ul[1]-ur[1]))
    # angle along lower part of stream
    th_b = np.arctan((bl[0]-br[0])/(bl[1]-br[1]))
    # compute correction by taking cosine of mean offset angle
    angle_correction = np.cos(np.mean(np.array([th_u, th_b])))

    return angle_correction


def get_frame_IDs(objs_archive, start, end, every):
    """Returns list of IDs of objects in each frame"""
    # TODO: fix bug when every != 1
    # initializes dictionary of IDs for each frame
    frame_IDs = {}
    for f in range(start, end, every):
        frame_IDs[f] = []
    # loads IDs of objects found in each frame
    for ID in objs_archive.keys():
        obj = objs_archive[ID]
        frames = obj.get_props('frame')
        for f in frames:
            frame_IDs[f] += [ID]

    return frame_IDs


def get_points(Npoints=1,im=None):
    """ Alter the built in ginput function in matplotlib.pyplot for custom use.
    This version switches the function of the left and right mouse buttons so
    that the user can pan/zoom without adding points. NOTE: the left mouse
    button still removes existing points.
    INPUT:
        Npoints = int - number of points to get from user clicks.
    OUTPUT:
        pp = list of tuples of (x,y) coordinates on image
    """
    if im is not None:
        plt.imshow(im)
        plt.axis('image')

    pp = plt.ginput(n=Npoints,mouse_add=3, mouse_pop=1, mouse_stop=2,
                    timeout=0)
    return pp


def highlight_obj(frame, bkgd, th_lo, th_hi, min_size, selem,
                     width_border=2, ret_all_steps=False):
    """Highlights objects in frame."""
    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(bkgd, frame)
    # based on assumption that objects are darker than bkgd, ignore all 
    # pixels that are brighter than the background by setting to zero
    im_diff[frame > bkgd] = 0

    ################# HYSTERESIS THRESHOLD AND LOW MIN SIZE ###################
    # thresholds image to become black-and-white
    thresh_bw = hysteresis_threshold(im_diff, th_lo, th_hi)
    thresh_bw = basic.cvify(thresh_bw)
    # smooths out thresholded image
    opened = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, selem)
    # removes small objects
    small_obj_rm = remove_small_objects(opened, min_size_hyst)
    # fills in holes, including those that might be cut off at border
    highlighted_objs = frame_and_fill(small_obj_rm, width_border)

    if ret_all_steps:
        return im_diff, thresh_bw, opened, \
                small_obj_rm, highlighted_objs
    else:
        return highlighted_objs
    
    
def highlight_obj_hyst(frame, bkgd, th_lo, th_hi, width_border, selem,
                          min_size, ret_all_steps=False):
    """
    Version of highlight_obj() that uses a hysteresis filter.
    """
    assert (len(frame.shape) == 2) and (len(bkgd.shape) == 2), \
        'improc.highlight_obj_hyst() only accepts 2D frames.'
    assert th_lo < th_hi, \
        'In improc.highlight_objs_hyst(), low threshold must be lower.'

    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(bkgd, frame)
    # thresholds image to become black-and-white
    # thresh_bw = skimage.filters.apply_hysteresis_threshold(\
                        # im_diff, th_lo, th_hi)
    thresh_bw = hysteresis_threshold(im_diff, th_lo, th_hi)

    # smooths out thresholded image
    closed_bw = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, selem)
    # removes small objects
    obj_bw = remove_small_objects(closed_bw, min_size)
    # fills in holes, including those that might be cut off at border
    obj = frame_and_fill(obj_part_bw, width_border)

    # returns intermediate steps if requeseted.
    if ret_all_steps:
        return im_diff, thresh_bw, closed_bw, obj_bw, \
                obj
    else:
        return obj


def highlight_obj_hyst_thresh(frame, bkgd, th, th_lo, th_hi, min_size_hyst,
                                 min_size_th, width_border, selem, mask_data,
                                 ret_all_steps=False):
    """
    Version of highlight_obj() that first performs a low threshold and
    high minimum size to get faint, large objects, and then performs a higher
    hysteresis threshold with a low minimum size to get distinct, small
    objects.

    Only accepts 2D frames.
    """
    assert (len(frame.shape) == 2) and (len(bkgd.shape) == 2), \
        'improc.highlight_obj_hyst_thresh() only accepts 2D frames.'
    assert th_lo < th_hi, \
        'In improc.highlight_objs_hyst_thresh(), low threshold must be lower.'

    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(bkgd, frame)
    # based on assumption that objects are darker than bkgd, ignore all 
    # pixels that are brighter than the background by setting to zero
    im_diff[frame > bkgd] = 0

    ##################### THRESHOLD AND HIGH MIN SIZE #########################
    # thresholds image to become black-and-white
    thresh_bw_1 = thresh_im(im_diff, th)
    # smooths out thresholded image
    closed_bw_1 = cv2.morphologyEx(thresh_bw_1, cv2.MORPH_OPEN, selem)
    # removes small objects
    obj_bw_1 = remove_small_objects(closed_bw_1, min_size_th)
    # fills enclosed holes with white, but leaves open holes black
    obj_1 = basic.fill_holes(obj_bw_1)

    ################# HYSTERESIS THRESHOLD AND LOW MIN SIZE ###################
    # thresholds image to become black-and-white
    # thresh_bw_2 = skimage.filters.apply_hysteresis_threshold(\
    #                     im_diff, th_lo, th_hi)
    thresh_bw_2 = hysteresis_threshold(im_diff, th_lo, th_hi)
    thresh_bw_2 = basic.cvify(thresh_bw_2)
    # smooths out thresholded image
    closed_bw_2 = cv2.morphologyEx(thresh_bw_2, cv2.MORPH_OPEN, selem)
    # removes small objects
    obj_bw_2 = remove_small_objects(closed_bw_2, min_size_hyst)
    # fills in holes, including those that might be cut off at border
    obj_2 = frame_and_fill(obj_bw_2, width_border)

    # merges images to create final image and masks result
    obj = np.logical_or(obj_1, obj_2)
    # TODO determine if I want to mask here
    #if mask_data is not None:
    #    obj = np.logical_and(obj, mask_data['mask'])

    # returns intermediate steps if requeseted.
    if ret_all_steps:
        return im_diff, thresh_bw_1, obj_1, thresh_bw_2, \
                obj_2, obj
    else:
        return obj


def highlight_obj_thresh(frame, bkgd, thresh, width_border, selem, min_size,
                     ret_all_steps=False):
    """
    Highlights objects (regions of different brightness) with white and
    turns background black. Ignores edges of the frame.
    Only accepts 2D frames.
    """
    assert (len(frame.shape) == 2) and (len(bkgd.shape) == 2), \
        'improc.highlight_obj() only accepts 2D frames.'

    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(bkgd, frame)
    # thresholds image to become black-and-white
    thresh_bw = thresh_im(im_diff, thresh)
    # smooths out thresholded image
    closed_bw = cv2.morphologyEx(thresh_bw, cv2.MORPH_OPEN, selem)
    # removes small objects
    obj_bw = remove_small_objects(closed_bw, min_size)
    # fills in holes, including those that might be cut off at border
    obj = frame_and_fill(obj_part_filled, width_border)

    # returns intermediate steps if requested.
    if ret_all_steps:
        return im_diff, thresh_bw, closed_bw, obj_bw, obj
    else:
        return obj


def hysteresis_threshold(im, th_lo, th_hi):
    """
    Applies a hysteresis threshold using only OpenCV methods to replace the
    method from scikit-image, skimage.filters.hysteresis_threshold.

    Lightly adapted from:
    https://stackoverflow.com/questions/47311595/opencv-hysteresis-thresholding-implementation
    """
    _, thresh_upper = cv2.threshold(im, th_hi, 128, cv2.THRESH_BINARY)
    _, thresh_lower = cv2.threshold(im, th_lo, 128, cv2.THRESH_BINARY)

    # finds the contours to get the seed from which to start the floodfill
    # the number of outputs depends on the major version of OpenCV
    v_cv2 = int(cv2.__version__[0])
    if v_cv2 == 3:
        _, cnts_upper, _ = cv2.findContours(thresh_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif v_cv2 == 4:
        cnts_upper, _ = cv2.findContours(thresh_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        print('OpenCV version is not 3 or 4--behavior of cv2.findContours() unknown')

    # Makes brighter the regions that contain a seed
    for cnt in cnts_upper:
        # C++ implementation
        # cv2.floodFill(threshLower, cnt[0], 255, 0, 2, 2, CV_FLOODFILL_FIXED_RANGE)
        # Python implementation
        cv2.floodFill(thresh_lower, None, tuple(cnt[0][0]), 255, 2, 2, cv2.FLOODFILL_FIXED_RANGE)

    # thresholds the image again to make black the unfilled regions
    _, im_thresh = cv2.threshold(thresh_lower, 200, 255, cv2.THRESH_BINARY)

    return im_thresh


def is_color(im):
    """Returns True if the image is a color image (3 channels) and false if not."""
    return len(im.shape) == 3


def is_on_border(bbox, im, width_border):
    """
    Checks if object is on the border of the frame.

    bbox is (min_row, min_col, max_row, max_col). Pixels belonging to the
    bounding box are in the half-open interval [min_row; max_row) and
    [min_col; max_col).
    """
    min_col = bbox[1]
    max_col = bbox[3]
    width_im = im.shape[1]
    if (min_col <= width_border) or (max_col >= width_im - width_border):
        return True
    else:
        return False


def lost_obj(centroid_pred, frame_labeled, ID, objs_archive):
    """
    Determines if object is "lost" and thus not worth tracking anymore in the
    case that the object is not detected in a frame.

    The object is "lost" if:
        1) the predicted location of its centroid is out of the boundaries of
        the frame
        2) the object has only been spotted once (so we don't have a good
        estimate of where the centroid would be so it's possible it is out of
        bounds)

    Parameters
    ----------
    centroid_pred : 2-tuple of floats
        (row, col) coordinates predicted for centroid of object (see
        TrackedObj.predict_centroid())
    frame_labeled : (M x N) numpy array of uint8
        Frame whose pixel values are the ID numbers of the objects where the
        pixels are located (0 if not part of a object)
    ID : int
        ID number of object assigned in assign_obj()
    objs_archive : dictionary of TrackedObj objects
        Dictionary of objects registered by ID number

    Returns
    -------
    lost : bool
        If True, object is deemed lost. Otherwise deemed detectable in future
        frames.
    """
    lost = out_of_bounds(centroid_pred, frame_labeled.shape) or \
                    (len(objs_archive[ID].get_props('frame')) < 2)
    return lost


def mask_im(im, mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # Applies mask depending on dimensions of image
    tmp = np.shape(im)
    im_masked = np.zeros_like(im)
    if len(tmp) == 3:
        for i in range(3):
            im_masked[:,:,i] = mask*im[:,:,i]
    else:
        im_masked = im*mask

    return im_masked


def measure_labeled_im_width(im_labeled, um_per_pix):
    """
    j
    """
    # Count labeled pixels in each column (roughly stream width)
    num_labeled_pixels = np.sum(im_labeled, axis=0)
    # Use coordinates of corners of labeled region to correct for oblique angle
    # (assume flat sides) and convert from pixels to um
    angle_correction = get_angle_correction(im_labeled)
    stream_width_arr = num_labeled_pixels*um_per_pix*angle_correction
    # remove columns without any stream (e.g., in case of masking)
    stream_width_arr = stream_width_arr[stream_width_arr > 0]
    # get mean and standard deviation
    mean = np.mean(stream_width_arr)
    std = np.std(stream_width_arr)

    return mean, std

    
def med_alg(cap, frame_trio):
    """
    Same as med_alg_thread but using VideoCapture obj instead of thread.
    Slower, but more reliable and predictable operation.
    OBSOLETE: doesn't work in some edge cases!!! (thanks koe for the catch!)
    """
    # reads frame from file video stream
    ret, frame = cap.read()
    frame = frame.astype(float)
    # adds frame to list if three frames not collected yet
    if len(frame_trio) < 3:
        frame_trio += [frame]
    # if three frames collected, takes their median and sets that as the new frame in the last
    else:
        stacked_arr = np.stack(tuple(frame_trio), axis=0)
        bkgd_med = np.median(stacked_arr, axis=0)
        frame_trio = [bkgd_med]

    return ret, (frame_trio,)


def med_alg_thread(fvs, frame_trio):
    """
    Performs repeated step for algorithm to compute the pixel-wise median of
    the frames of a video. It loads two frames along with the current median
    into a "trio" and then computes the pixel-wise median of those three
    frames. This can be shown to give the overall pixel-wise median of a
    series of frames without requiring all frames to be stored in memory
    simultaneously.

    Parameters
    ----------
    fvs : FileVideoStream object
        File video stream that manages threading used to load and queue frames.
        See the FileVideoStream class in classes.py for definition and methods.
        Construct and initiate: FileVideoStream(<str of vid filepath>).start()
    frame_trio : list
        List of numpy arrays of frames. The first element is always the median.
        The list may have up to three elements, the second and third being
        frames in the queue for computing the median.

    Returns
    -------
    frame_trio : list
        Same as input frame_trio, but either with the loaded frame appended or
        the median taken and only the median frame remaining.

    """
    # reads frame from file video stream
    frame = fvs.read().astype(float)
    # adds frame to list if three frames not collected yet
    if len(frame_trio) < 3:
        frame_trio += [frame]
    # if three frames collected, takes their median and sets that as the new frame in the last
    else:
        stacked_arr = np.stack(tuple(frame_trio), axis=0)
        bkgd_med = np.median(stacked_arr, axis=0)
        frame_trio = [bkgd_med]

    return (frame_trio,)


def obj_d_mat(objs_prev, objs_curr, d_fn, d_fn_kwargs):
    """
    Computes the distance matrix of distances between each pair of previous
    and current objects based on the metric function given.
    Used in `assign_obj`.
    
    Parameters
    ----------
    objs_prev : list of dictionaries
        List of dictionaries of properties of objects in previous frame
        *Dictionaries must included 'centroid' property
    objs_curr : list of dictionaries
        Same as objs_prev but for objects in current frame
    d_fn : function
        Function to compute distance between objects.
        *Must have arguments (obj1, obj2, **d_fn_kwargs), where obj1 and obj2
        are dictionaries of the format in objs_prev and objs_curr
    d_fn_kwargs : dictionary
        Keyword arguments for d_fn in a dictionary of the format
        d_fn_kwargs['keyword_arg'] = value
        
    Returns
    -------
    d_mat : (M x N) numpy array
        Array of distances between objects in the previous frame (indexed by row)
        and objects in the current frame (indexed by column)
    """
    # gets dimensions of distance matrix
    M = len(objs_prev)
    N = len(objs_curr)
    
    # creates blank distance matrix
    d_mat = np.zeros([M, N])
    
    # computes distance between each pair of previous and current objects
    for i, obj_prev in enumerate(objs_prev):
        for j, obj_curr in enumerate(objs_curr):
            d_mat[i,j] = d_fn(obj_prev, obj_curr, **d_fn_kwargs)

    return d_mat


def one_2_uint8(im, copy=True):
    """
    Returns a copy of an image scaled from 0 to 1 as an image of uint8's scaled
    from 0-255.

    Parameters
    ----------
    im : 2D or 3D array of floats
        Image with pixel intensities measured from 0 to 1 as floats
    copy : bool, optional
        If True, image will be copied first

    Returns
    -------
    im_uint8 : 2D or 3D array of uint8's
        Image with pixel intensities measured from 0 to 255 as uint8's
    """
    if copy:
        im = np.copy(im)
    return (255*im).astype('uint8')


def out_of_bounds(pt, shape):
    """
    Returns True if point is in the bounds given by shape, False if not.

    Parameters
    ----------
    pt : 2-tuple of ints
        Point to check if out of bounds.
    shape : 2-tuple of ints
        Bounds (assuming from 0 to the values given in this tuple).

    Returns :
    out : bool
        True if out of bounds, False if in bounds.
    """
    # in bounds
    if pt[0] >= 0 and pt[0] < shape[0] and pt[1] >= 0 and pt[1] < shape[1]:
        return False
    # otherwise, out of bounds
    else:
        return True


def prep_for_mpl(im):
    """
    Prepares an image for display in matplotlib's imshow() method.

    Parameters
    ----------
    im : (M x N x 3) or (M x N) numpy array of uint8 or float
        Image to convert.

    Returns
    -------
    im_p : same dims as im, numpy array of uint8
        Image prepared for matplotlib's imshow()

    """
    if is_color(im):
        im_p = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im_p = np.copy(im)
    im_p = 255.0 / np.max(im_p) * im_p
    im_p = im_p.astype('uint8')

    return im_p


def proc_frames(cap, alg, args, num_frames=100, report_freq=10):
    """
    Processes frames without using threading due to unpredictable outcomes.
    """
    # initializes result by applying algorithm to first frame
    ret, result = alg(cap, *args)
    # initializes counter at 3 because the next computation will be the 3rd
    # (args is first, result is second)
    ctr = 3
    # loops through frames performing algorithm to process them
    while ret:

        # computes algorithmic step
        ret, result = alg(cap, *result)

        # reports counter and progress
        ctr += 1
        if ctr > num_frames:
            break
        if (ctr % report_freq) == 0:
            print('Completed {0:d} frames of {1:d}.'.format(ctr, num_frames))

    # when everything is done, releases the capture
    cap.release()

    return result


def proc_frames_thread(fvs, alg, args, num_frames=100):
    """
    Processes frames using threading for efficiency. Applies given algorithm
    upon loading each frame.

    NOTE: the FileVideoStream object fvs must be stopped before running this.
    Otherwise it will not be able to start.

    Parameters
    ----------
    fvs : FileVideoStream object
        File video stream that manages threading used to load and queue frames.
        See the FileVideoStream class in classes.py for definition and methods.
        Construct and initiate: FileVideoStream(<str of vid filepath>).start()
    alg : function
        Algorithmic step to apply to each new frame. Also takes in args. This
        function must return an array of floats. It will not load if it is int.
    args : tuple
        Additional arguments besides the FileVideoStream required by alg.
    num_frames : int, optional
        Number of frames to process. The default is 100.

    Returns
    -------
    result : object
        Output of repeated application of the algorithmic step alg to each
        frame in the video. Type is determined by output of alg.

    """
    print('started proc_frames_thread')
    # initializes result by applying algorithm to first frame
    result = alg(fvs, *args)
    # initializes counter at 3 because the next computation will be the 3rd (args is first, result is second)
    ctr = 3
    # loops through frames performing algorithm to process them
    while fvs.more():

        # computes algorithmic step
        result = alg(fvs, *result)

        # reports counter and progress
        ctr += 1
        if ctr > num_frames:
            break
        if (ctr % 10) == 0:
            print('Completed {0:d} frames of {1:d}.'.format(ctr, num_frames))

    # stops file video stream
    fvs.stop()

    return result


def proc_im_seq(im_path_list, proc_fn, params, columns=None):
    """
    Processes a sequence of images with the given function and returns the
    results. Images are provided as filepaths to images, which are loaded (and
    possibly copied before analysis to preserve the image).

    Parameters:
        im_path_list : array-like
            Sequence of filepaths to the images to be processed
        proc_fn : function handle
            Handle of function to use to process image sequence
        params : list
            List of parameters to plug into the processing function
        columns : array-like, optional
            Results from image processing are saved to this dataframe

    Returns:
        output : list (optionally, Pandas DataFrame if columns given)
            Results from image processing
    """
    # Initialize list to store results from image processing
    output = []
    # Process each image in sequence.
    for im_path in im_path_list:
        print("Begin processing {im_path}.".format(im_path=im_path))
        im = plt.imread(im_path)
        output += [proc_fn(im, params)]
    # If columns provided, convert list into a dataframe
    if columns:
        output = pd.DataFrame(output, columns=columns)

    return output


def region_props(frame_bw, n_frame=-1, width_border=5):
    """
    Computes properties of objects in a binarized image using OpenCV.
    """
    frame_bw = basic.cvify(frame_bw)
    return region_props_connected(frame_bw, n_frame=n_frame, width_border=width_border)


def region_props_connected(frame_bw, n_frame=-1, width_border=5):
    """
    Computes properties of objects in a binarized image that would otherwise be
    provided by region_props using an OpenCV hack.

    This version is based on connectedComponentsWithStats.
    """
    # identifies the different objects in the frame
    num_labels, frame_labeled, stats, centroids = cv2.connectedComponentsWithStats(frame_bw)
    # creates dictionaries of properties for each object
    objs_curr = []
    # records stats of each labeled object; skips 0-label (background)
    for i in range(1, num_labels):
        # creates dictionary of object properties for one frame, which
        # can be merged to a Tracked object
        obj = {}
        # switches default (x,y) -> (row, col)
        obj['centroid'] = centroids[i][::-1]
        obj['area'] = stats[i, cv2.CC_STAT_AREA]
        row_min = stats[i, cv2.CC_STAT_TOP]
        col_min = stats[i, cv2.CC_STAT_LEFT]
        row_max = row_min + stats[i, cv2.CC_STAT_HEIGHT]
        col_max = col_min + stats[i, cv2.CC_STAT_WIDTH]
        bbox = (row_min, col_min, row_max, col_max)
        obj['bbox'] = bbox

        if n_frame >= 0:
            obj['frame'] = n_frame

        obj['on border'] = is_on_border(bbox,
              frame_labeled, width_border)
        # adds dictionary for this object to list of objects in current frame
        objs_curr += [obj]

    return objs_curr


def region_props_find(frame_bw, n_frame=-1, width_border=5, ellipse=True):
    """
    Computes properties of objects in a binarized image that would otherwise be
    provided by region_props using an OpenCV hack.

    This version is based on findContours.
    """
    # computes contours of all objects in image
    cnts, _ = cv2.findContours(frame_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # creates dictionaries of properties for each object
    objs = []

    # records stats of each labeled object; skips 0-label (background)
    for i, cnt in enumerate(cnts):
        # creates dictionary of object properties for one frame, which
        # can be merged to a TrackedObj object
        obj = {}

        # computes moments of contour
        M = cv2.moments(cnt)

        # computes centroid
        #https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        obj['centroid'] = (cy, cx)

        # computes number of pixels in object (area)
        #https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html
        mask = np.zeros(frame_bw.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        # records length of list of nonzero pixels
        # TODO -- VALIDATE! Doesn't seem to work properly for counting pixels
        num_pixels = len(cv2.findNonZero(mask))
        obj['area'] = num_pixels

        # computes bounding box
        col_min, row_min, w, h = cv2.boundingRect(cnt)
        row_max = row_min + h
        col_max = col_min + w
        bbox = (row_min, col_min, row_max, col_max)
        obj['bbox'] = bbox

        # fits ellipse to compute major and minor axes, orientation; needs at
        # least 5 points to fit ellipse
        if ellipse:
            # if enough points to fit ellipse, let OpenCV take care of the props
            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # if fewer than 5 points, ad hoc estimates
            else:
                MA = max(w, h)
                ma = min(w, h)
                angle = 0
            # stores results
            obj['orientation'] = angle
            obj['major axis'] = MA
            obj['minor axis'] = ma

        # saves frame number
        if n_frame >= 0:
            obj['frame'] = n_frame

        # checks if object is on the border of the frame
        obj['on border'] = is_on_border(bbox,
              frame_bw, width_border)

        # adds dictionary for this object to list of objects in current frame
        objs += [obj]

    return objs


def remove_small_objects(im, min_size):
    """
    Removes small objects in an image.
    Uses the findContours version because the tests suggest that it is faster.
    """
    return remove_small_objects_find(im, min_size)


def remove_small_objects_find(im, min_size):
    """
    Removes small objects in image with OpenCV's findContours.
    Uses OpenCV to replicate `skimage.morphology.remove_small_objects`.

    Appears to be faster than with connectedComponentsWithStats (see
    compare_props_finders.py).

    Based on response from nathancy @
    https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
    """
    # Filter using contour area and remove small noise
    result = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ver = int(cv2.__version__[0])
    if ver == 3:
        _, cnts, _ = result
    else:
        cnts, _ = result
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_size:
            cv2.drawContours(im, [c], -1, (0,0,0), -1)

    return im


def remove_small_objects_connected(im, min_size):
    """
    Removes small objects in image based on number of pixels in the object.
    Uses OpenCV to replicate `skimage.morphology.remove_small_objects`.
    Uses cv2.connectedComponentsWithStats instead of cv2.findContours.

    Appears to be slower than with findContours (see compare_props_finders.py).

    Parameters
    ----------
    im : numpy array of uint8s
        image from which to remove small objects
    """
    # formats image for OpenCV
    im = basic.cvify(im)
    # computes maximum value, casting to uint8 type for OpenCV functions
    max_val = np.max(im)
    # finds and labels objects in image--note im_labeled is int32 type
    n_labels, im_labeled, stats, _ = cv2.connectedComponentsWithStats(im)
    # loops through non-zero labels (i.e., objects--bkgd is labeled 0)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            im_labeled[np.where(im_labeled==i)] = 0
        else:
            im_labeled[np.where(im_labeled==i)] = max_val

    return im_labeled.astype('uint8')


def scale_by_brightfield(im, bf):
    """
    scale pixels by value in brightfield

    Parameters:
        im : array of floats or ints
            Image to scale
        bf : array of floats or ints
            Brightfield image for scaling given image

    Returns:
        im_scaled : array of uint8
            Image scaled by brightfield image
    """
    # convert to intensity map scaled to 1.0
    bf_1 = bf.astype(float) / 255.0
    # scale by bright field
    im_scaled = np.divide(im,bf_1)
    # rescale result to have a max pixel value of 255
    im_scaled *= 255.0/np.max(im_scaled)
    # change type to uint8
    im_scaled = im_scaled.astype('uint8')

    return im_scaled


def thresh_im(im, thresh=-1, c=5):
    """
        Applies a threshold to the image and returns black-and-white result.
    c : int
        channel to threshold, default 5 indicates threshold brightest channel
    Modified from ImageProcessingFunctions.py
    """
    n_dims = len(im.shape)
    if n_dims == 3:
        if c > n_dims:
            c_brightest = -1
            pix_val_brightest = -1
            for i in range(n_dims):
                curr_brightest = np.max(im[:,:,i])
                if curr_brightest > pix_val_brightest:
                    c_brightest = i
                    pix_val_brightest = curr_brightest
        else:
            c_brightest = c
        im = np.copy(im[:,:,c_brightest])
    if thresh == -1:
        # Otsu's thresholding
        ret, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret, thresh_im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

    return thresh_im

def track_obj(track_obj_method, track_kwargs, highlight_kwargs, assignobjs_kwargs, ret_IDs=False):
    objs_archive = track_obj_method(track_kwargs, highlight_kwargs, assignobjs_kwargs)

    # only returns IDs for each frame if requested
    # note: assumes objs_archive is aligned with start/end/every
    if ret_IDs:
        frame_IDs = get_frame_IDs(objs_archive, track_kwargs['start'], track_kwargs['end'], track_kwargs['every'])
        return objs_archive, frame_IDs
    else:
        return objs_archive

def track_obj_py(track_kwargs, highlight_kwargs, assignobjs_kwargs):
    vid_path = track_kwargs['vid_path']
    highlight_obj_method = track_kwargs['highlight_obj_method']
    print_freq = track_kwargs['print_freq']
    start = track_kwargs['start']
    end = track_kwargs['end']
    every = track_kwargs['every']
    row_lo = assignobjs_kwargs['row_lo']
    row_hi = assignobjs_kwargs['row_hi']

    """
    ***TODO: install and implement decord VideoReader to speed up loading of
    frames: https://github.com/dmlc/decord***
    """
    # initializes ordered dictionary of object data from past frames and archive of all data
    objs_prev = OrderedDict()
    objs_archive = {}
    # initializes counter of current object label (0-indexed)
    ID_curr = 0
    # chooses end frame to be last frame if given as -1
    if end == -1:
        end = basic.count_frames(vid_path)

    # loops through frames of video
    for f in range(start, end, every):

        # loads frame from video file
        frame, _ = basic.load_frame(vid_path, f)

        # crop the frame (only height is cropped in the current version)
        frame = frame[row_lo:row_hi, 0:frame.shape[1]]

        # extracts value channel of frame--including selem ruins segmentation
        val = basic.get_val_channel(frame)

        # highlights objects in the given frame
        objs_bw = highlight_obj_method(val, track_kwargs['bkgd'], **highlight_kwargs)

        # finds objects and assigns IDs to track them, saving to archive
        ID_curr = assign_obj(objs_bw, f, objs_prev, objs_archive, ID_curr, **assignobjs_kwargs)

        if (f % print_freq*every) == 0:
            print('Processed frame {0:d} of range {1:d}:{2:d}:{3:d}.' \
                  .format(f, start, every, end))
        # a5 = time.time()
        # print('5 {0:f} ms.'.format(1000*(a5-a4)))

    return objs_archive

def track_obj_cvvidproc(track_kwargs, highlight_kwargs, assignobjs_kwargs):
    highlightpack = cvvidproc.HighlightObjectsPack(
        background=track_kwargs['bkgd'],
        struct_element=highlight_kwargs['selem'],
        threshold=highlight_kwargs['th'],
        threshold_lo=highlight_kwargs['th_lo'],
        threshold_hi=highlight_kwargs['th_hi'],
        min_size_hyst=highlight_kwargs['min_size_hyst'],
        min_size_threshold=highlight_kwargs['min_size_th'],
        width_border=highlight_kwargs['width_border'])

    assignpack = cvvidproc.AssignObjectsPack(
        track_kwargs['assign_obj_method'],     # pass in function name as functor (not string)
        assignobjs_kwargs)

    # fields not defined will be defaulted
    trackpack = cvvidproc.VidObjectTrackPack(
        vid_path=track_kwargs['vid_path'],
        highlightobjs_pack=highlightpack,
        assignobjs_pack=assignpack,
        frame_limit=track_kwargs['end']-track_kwargs['start'],
        grayscale=True,
        crop_y=assignobjs_kwargs['row_lo'],
        crop_height=assignobjs_kwargs['row_hi']-assignobjs_kwargs['row_lo'],
        print_timing_report=True)

    print('tracking objects...')
    start_time = time.time()

    objs_archive = cvvidproc.Trackobjects(trackpack)

    end_time = time.time()
    print('Tracked ({0:f} object(s); {1:f} s)'.format(len(objs_archive), end_time - start_time))

    return objs_archive



