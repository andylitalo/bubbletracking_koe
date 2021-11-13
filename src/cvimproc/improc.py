"""
improc.py contains definitions of methods for image-processing
both for pure-Python analysis and for CvVidProc analysis using
only OpenCV.

TODO: move OpenCV methods to an archive since they were transferred
to C++ in CvVidProc.
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
import skimage.measure # imported since assign_objects is still in Python

# imports custom image processing libraries
import cvimproc.mask as mask
import cvimproc.basic as basic

# imports general custom libraries
import genl.fn as fn
import genl.geo as geo
from genl.conversions import *
import cvimproc.pltim as pltim

# imports custom classes (with reload clauses)
from classes.classes import TrackedObject

# import custom opencv video processing methods
import cvvidproc

############################# METHOD DEFINITIONS ##############################


def appears_in_consec_frames(obj, n_consec=1):
    """Returns True if object appears in consecutive frames."""
    return len(np.where( \
                np.diff(obj.get_props('frame'))==1)[0]) >= n_consec


def assign_objects(bw_frame, frames_processed, objects_prev, 
                    objects_archive, next_ID, kwargs):
    """
    Assigns objects with unique IDs to the labeled objects on the video
    frame provided. This method is used on a single frame in the context of
    processing an entire video and uses information from the previous frame.
    Inspired by and partially copied from PyImageSearch [1].

    Only registers objects above the minimum registration size.
    Once registered, an object no longer needs to remain above the 
    area threshold.

    Updates objects_prev and objects_archive in place.

    ***OpenCV-compatible version: labels frame instead of receiving labeled
    frame by using cv2.connectedComponentsWithStats.

    Parameters
    ----------
    bw_frame : (M x N) numpy array of uint8
        Binarized video frame
    frames_processed : int
        Number of the current frame in video
        # TODO change back to `f` in CvVidProc since != frames processed if every != 1 or start != 0
    objects_prev : Dictionary of dictionaries
        Contains dictionaries of properties of objects from the previous frame
        indexed by ID number
    objects_archive : dictionary of TrackedObj objects
        Dictionary of objects from all previous frames
    next_ID : int
        Next ID number to be assigned (increasing order)
    kwargs : dictionary
        fps : float
            Frames per second of video
        d_fn : functor
            Function for computing the distance between centroids of objects
        d_fn_kwargs : dictionary
            Dictionary of keyword arguments for `d_fn`. The keyword arguments 
            are all those that follow after `object1` and `object2` (the first
            two arguments--they may go by different names)
        width_border : int
            Number of pixels around perimeter that count as "the border" of the
            image. Used in `frame_and_fill`.
        min_size_reg : int
            Minimum number of pixels an object must have to be registered
        row_lo : int
            Row of lower inner wall--currently not implemented
        row_hi : int
            Row of upper inner wall--currently not implemented
        remember_objects : bool
            If True, code will predict centroid location after an object 
            disappears using its previous velocity
        ellipse : bool
            If True, region_props will fit an ellipse to the object to compute
            major axis, minor axis, and orientation
        ObjectClass : class
            Class in which to instantiate objects
        object_kwargs : dictionary
            Keyword arguments needed for the instantiation of an object from 
            ObjectClass

    Returns
    -------
    next_ID : int
        Updated value of next ID number to assign.

    References
    ----------
    .. [1] https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """

    ### PARSE ARGS ###
    # TODO remove this defn
    f = frames_processed
    # extracts keyword arguments for the assign method (not explicitly
    # requested args to match syntax of CvVidProc library)
    fps = kwargs['fps']
    d_fn = kwargs['d_fn']
    d_fn_kwargs = kwargs['d_fn_kwargs']
    width_border = kwargs['width_border']
    min_size_reg = kwargs['min_size_reg']
    row_lo = kwargs['row_lo']
    row_hi = kwargs['row_hi']
    remember_objects = kwargs['remember_objects']
    ellipse = kwargs['ellipse']
    ObjectClass = kwargs['ObjectClass']
    object_kwargs = kwargs['object_kwargs']
    filter_fn = kwargs['filter_fn']
    filter_kwargs = kwargs['filter_kwargs']

    ### MEASURES PROPERTIES ###
    # computes frame dimesions
    frame_dim = bw_frame.shape
    # measures region props of each object in image
    objects_curr = region_props(bw_frame, n_frame=f, 
                            width_border=width_border, ellipse=ellipse,
                            filter_fn=filter_fn, filter_kwargs=filter_kwargs)

    ### HANDLES CASES WITH EMPTY FRAME ###
    # if no objects seen in previous frame, assigns objects in current frame
    # to new IDs
    if len(objects_prev) == 0:
        for i in range(len(objects_curr)):
            objects_prev[next_ID] = objects_curr[i]
            next_ID += 1

    # if no objects in current frame, removes objects from dictionary of
    # objects in the previous frame
    elif len(objects_curr) == 0:
        for ID in list(objects_prev.keys()):
            # predicts the next centroid for the object based on previous velocity
            centroid_pred = objects_archive[ID].predict_centroid(f)
            # if the most recent (possibly predicted) centroid is out of bounds,
            # or if our prediction comes from a single data point (so the
            # velocity is uncertain),
            # then the object is deleted from the dictionary
            if lost_obj(centroid_pred, bw_frame, ID, objects_archive) or \
                (not remember_objects):
                del objects_prev[ID]
            # otherwise, predicts next centroid, keeping other props the same
            else:
                objects_prev[ID]['frame'] = f
                objects_prev[ID]['centroid'] = centroid_pred


    ### HANDLES CASE OF TWO FRAMES WITH OBJECTS ###
    # otherwise, assigns objects in current frames to previous objects based
    # on provided distance function
    else:
        # grabs the set of object IDs from the previous frame
        IDs = list(objects_prev.keys())
        # computes M x N matrix of distances (M = # objects in previous frame,
        # N = # objects in current frame)
        d_mat = obj_d_mat(list(objects_prev.values()),
                             objects_curr, d_fn, d_fn_kwargs)

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
            d_longest = np.linalg.norm(bw_frame.shape)
            if d_mat[row, col] > d_longest:
                continue

            # otherwise, grabs the object ID for the current row,
            # set its new centroid, and resets the disappeared
            # counter
            ID = IDs[row]
            objects_prev[ID] = objects_curr[col]
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
            centroid_pred = objects_archive[ID].predict_centroid(f)
            # deletes object if centroid is out of bounds or if prediction is
            # based on just 1 data point (so velocity is uncertain)
            if lost_obj(centroid_pred, bw_frame, ID, objects_archive) or (not remember_objects):
                del objects_prev[ID]
            # otherwise, predicts next centroid, keeping other props the same
            else:
                objects_prev[ID]['frame'] = f
                objects_prev[ID]['centroid'] = centroid_pred


        # registers each unregistered new input centroid as a object seen
        for col in cols_unused:
            # adds only objects above threshold
            if objects_curr[col]['area'] >= min_size_reg:
                objects_prev[next_ID] = objects_curr[col]
                next_ID += 1


    ### ARCHIVES OBJECTS ###
    # archives objects from this frame in order of increasing ID
    for ID in objects_prev.keys():
        # creates new ordered dictionary of objects if new object
        if ID == len(objects_archive):
            objects_archive[ID] = ObjectClass(ID, fps, frame_dim, 
                                                props_raw=objects_prev[ID], 
                                                **object_kwargs)
        elif ID < len(objects_archive):
            objects_archive[ID].add_props(objects_prev[ID])
        else:
            print('In assign_objects(), IDs looped out of order while saving to archive.')

    return next_ID


def bubble_distance_v(bubble1, bubble2, axis, row_lo, row_hi, 
                        v_max, v_interf, fps,
                        min_travel=0, upstream_penalty=1E5,
                        min_off_axis=4, off_axis_steepness=0.3,
                        alpha=1, beta=1):
    """
    Heuristic function for quantifying "distance" between objects in
    videos of bubbles in flow used for assigning consistent labels to
    objects during object-tracking.
    Incorporates predicted velocity of stream.

    Parameters
    ----------

    v_max [pix/s]
    Computes the distance between each pair of points in the two sets
    perpendicular to the axis. All inputs must be numpy arrays.
    Wiggle room gives forgiveness for a few pixels in case the bubble is
    stagnant but processing causes the centroid to move a little.
    
    This should never be greater than the length of the frame unless bubble2
    *definitely* does not belong to bubble1 (e.g., if bubble2 is upstream of bubble1).
    # TODO incorporate velocity profile more accurately into objective
    """
    # computes distance between the centroids of the two bubbles [row, col]
    c1 = np.array(bubble1['centroid'])
    c2 = np.array(bubble2['centroid'])
    diff = c2 - c1
    # computes components on and off axis
    comp, off_axis = geo.calc_comps(diff, axis)
    

    # computes average distance off central flow axis [pix]
    row_center = (row_lo + row_hi)/2
    origin = np.array([row_center, 0])
    rz1 = c1 - origin
    _, r1 = geo.calc_comps(rz1, axis)
    rz2 = c2 - origin
    _, r2 = geo.calc_comps(rz2, axis)
    r = (r1 + r2)/2
    # computes inner stream radius [pix]
    R = np.abs(row_lo - row_hi)
    # computes velocity assuming Poiseuille flow [pix/s]
    v = (v_max - v_interf)*(1 - (r/R)**2) + v_interf
    # time step per frame [s]
    dt = 1/fps
    # expected distance along projected axis [pix]
    comp_expected = v*dt
    

    # adds huge penalty if second bubble is upstream of first bubble and a
    # moderate penalty if it is off the axis or far from expected position
    d = alpha*off_axis + beta*np.abs((comp - comp_expected)/comp_expected) + \
        upstream_penalty*(comp < min_travel)

    return d


def bubble_distinction():
    """
    Builds on `bubble_distance_v()` by incorporating score for
    shape and size similarity in determining if something is a 
    bubble or not.

    Still prototyping this.

    Distinguishing features:
    - Distance off axis (already in bubble_distance_v)
    - Distance along axis relative to prediction based on 
    velocity (already in bubble_distance_v)
    - Shape similarity (aspect ratio, solidity, orientation)
    - Size similarity (area, bounding box width and height, major and minor axes)

    Can I tune a model to predict similarity based on these parameters?
    Which parameters do I already have access to?
    """

    pass


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


def compute_bkgd_med(vid_path, vid_is_grayscale=True, num_frames=100, 
			crop_x=0, crop_y=0, crop_width=0, crop_height=0,
			max_threads=-1):
    """Alias for `compute_bkgd_med_thread` since I haven't programmed an unthreaded version."""
    return  compute_bkgd_med_thread(vid_path, vid_is_grayscale, num_frames=num_frames, 
			crop_x=crop_x, crop_y=crop_y, crop_width=crop_width, crop_height=crop_height,
			max_threads=max_threads)


def compute_bkgd_med_thread(vid_path, vid_is_grayscale=True, num_frames=100, 
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
    Computes the distance between the centroids of the given objects
    using the Euclidean distance (L2-norm).
    
    Parameters
    ----------
    obj1, obj2 : dictionary
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


def d_off_flow(obj1, obj2, flow_dir, penalty=1E10):
    """
    Computes the component of the distance between the two objects 
    perpendicular to the axis.

    Parameters
    ----------
    obj1, obj2 : dictionary
        Dictionaries of object properties containing 'centroid'
    flow_dir : 2-tuple of floats
        Axis along which we expect motion (off-axis motion is lightly 
        penalized and upstream motion is harshly penalized).
        Format is (row, col)

    Returns
    -------
    d : float
        Distance between the centroids of object 1 and object 2
    """
    # extracts centroids of the two objects in (row, col) format [pixels]
    c1 = np.array(obj1['centroid'])
    c2 = np.array(obj2['centroid'])
    diff = c2 - c1
    # computes components on and off axis
    comp, off_axis = geo.calc_comps(diff, flow_dir)

    # adds penalty if object is "upstream"
    d = off_axis + penalty*(comp < 0)

    return d
   

def filter_sph(obj, min_size=12, min_solidity=0.9, max_angle=np.pi/10,
                aspect_ratio_sph=1.3):
    """
    Returns True if object is roughly spherical and the right size and False
    if not.

    Parameters
    ----------
    obj : TrackedObj
        Object to filter.
    
    Returns
    -------
    passed : bool
        True if passed filter (spherical and right size) and False if not.
    """
    # The following are "uninteresting":
    # small objects
    if obj['area'] <= min_size:
        return False
    # concave objects
    if obj['solidity'] <= min_solidity:
        return False
    # objects that are too oblong
    w = obj['bbox'][3] - obj['bbox'][1]
    h = obj['bbox'][2] - obj['bbox'][0]
    # objects that are not oriented along flow direction and elongated
    if (np.abs(obj['orientation'] + np.pi/2) > max_angle) and \
            (np.abs(obj['orientation'] - np.pi/2) > max_angle) and \
            (w / h > aspect_ratio_sph):
        return False

    return True


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
    im = mask.mask_image(im_filled, np.logical_not(mask_frame_sides))

    return im


def get_frame_IDs(objects_archive, start, end, every):
    """
    Returns list of ID numbers of the objects identified in each frame.
    
    Parameters
    ----------
    objects_archive : dictionary
        Dictionary of objects identified in a video labeled by ID number
    start, end, every : ints
        start = index of first frame to analyze; end = index of last frame
        to analyze; every = analyze every `every` frame (e.g., if every = 3,
        analyzes every 3rd frame)
    
    Returns
    -------
    frame_IDs : dictionary
        Dictionary indexed by frame number in the video. Each entry is
        a list of the ID numbers of the objects identified in that frame.
    """
    # initializes dictionary of IDs for each frame
    frame_IDs = {}
    for f in range(start, end, every):
        frame_IDs[f] = []
    # loads IDs of objects found in each frame
    for ID in objects_archive.keys():
        obj = objects_archive[ID]
        frames = obj.get_props('frame')
        for f in frames:
            frame_IDs[f] += [ID]

    return frame_IDs


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
    highlighted_objects = frame_and_fill(small_obj_rm, width_border)

    if ret_all_steps:
        return im_diff, thresh_bw, opened, \
                small_obj_rm, highlighted_objects
    else:
        return highlighted_objects


def highlight_obj_hyst_thresh(frame, bkgd, th, th_lo, th_hi, min_size_hyst,
                                 min_size_th, width_border, selem, mask_data,
                                 ret_all_steps=False, only_dark_obj=True):
    """
    Version of highlight_obj() that first performs a low threshold and
    high minimum size to get faint, large objects, and then performs a higher
    hysteresis threshold with a low minimum size to get distinct, small
    objects.

    Only accepts 2D frames.
    """
    # checks that the frames are 2D and that low threshold is lower than the high
    assert (len(frame.shape) == 2) and (len(bkgd.shape) == 2), \
        'improc.highlight_obj_hyst_thresh() only accepts 2D frames.'
    assert th_lo < th_hi, \
        'In improc.highlight_objects_hyst_thresh(), low threshold must be lower.'

    # masks inputs
    # TODO -- implement in CvVidProc
    if mask_data is not None:
        # computes minimum and maximum rows for object tracking computation
        row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
        # crops mask to same size as images and applies it
        frame = mask.mask_image(frame, mask_data['mask'][row_lo:row_hi, :])
        bkgd = mask.mask_image(bkgd, mask_data['mask'][row_lo:row_hi, :])
        
    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(bkgd, frame)
    # based on assumption that objects are darker than bkgd, ignore all 
    # pixels that are brighter than the background by setting to zero
    if only_dark_obj:
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
    # obj_2 = obj_bw_2 # TODO reinstate frame_and_fill
    # merges images to create final image and masks result
    obj = np.logical_or(obj_1, obj_2)

    # returns intermediate steps if requeseted.
    if ret_all_steps:
        return im_diff, thresh_bw_1, obj_1, thresh_bw_2, \
                obj_2, obj
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


def lost_obj(centroid_pred, frame_labeled, ID, objects_archive):
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
        ID number of object assigned in assign_objects()
    objects_archive : dictionary of TrackedObj objects
        Dictionary of objects registered by ID number

    Returns
    -------
    lost : bool
        If True, object is deemed lost. Otherwise deemed detectable in future
        frames.
    """
    lost = out_of_bounds(centroid_pred, frame_labeled.shape) or \
                    (len(objects_archive[ID].get_props('frame')) < 2)
    return lost


def obj_d_mat(objects_prev, objects_curr, d_fn, d_fn_kwargs):
    """
    Computes the distance matrix of distances between each pair of previous
    and current objects based on the metric function given.
    Used in `assign_objects`.
    
    Parameters
    ----------
    objects_prev : list of dictionaries
        List of dictionaries of properties of objects in previous frame
        *Dictionaries must include 'centroid' property
    objects_curr : list of dictionaries
        Same as objects_prev but for objects in current frame
    d_fn : function
        Function to compute distance between objects.
        *Must have arguments (obj1, obj2, **d_fn_kwargs), where obj1 and obj2
        are dictionaries of the format in objects_prev and objects_curr
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
    M = len(objects_prev)
    N = len(objects_curr)
    
    # creates blank distance matrix
    d_mat = np.zeros([M, N])
    
    # computes distance between each pair of previous and current objects
    for i, obj_prev in enumerate(objects_prev):
        for j, obj_curr in enumerate(objects_curr):
            d_mat[i,j] = d_fn(obj_prev, obj_curr, **d_fn_kwargs)

    return d_mat


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


def region_props(bw_frame, n_frame=-1, width_border=5, ellipse=False,
                filter_fn=None, filter_kwargs={}):
    """
    Computes properties of objects in a binarized image using OpenCV.
    """
    bw_frame = basic.cvify(bw_frame)
    # TODO -- clean this up
    try:
        import skimage.measure
        objects = region_props_skimage(bw_frame, n_frame=n_frame, 
                                    width_border=width_border)
    except:
        objects = region_props_find(bw_frame, n_frame=n_frame, 
                    width_border=width_border, ellipse=ellipse)

    # filters out objects with undesired properties if filter fn provided
    if filter_fn is not None:
        objects = [obj for obj in objects if filter_fn(obj, **filter_kwargs)]

    return objects


def region_props_connected(bw_frame, n_frame=-1, width_border=5):
    """
    Computes properties of objects in a binarized image that would otherwise be
    provided by region_props using an OpenCV hack.

    SLOW compared to region_props_find (see `tests/results.txt`)

    This version is based on connectedComponentsWithStats.
    """
    # identifies the different objects in the frame
    num_labels, frame_labeled, stats, centroids = cv2.connectedComponentsWithStats(bw_frame)
    # creates dictionaries of properties for each object
    objects_curr = []
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
        objects_curr += [obj]

    return objects_curr


def region_props_find(bw_frame, n_frame=-1, width_border=5, ellipse=True):
    """
    Computes properties of objects in a binarized image that would otherwise be
    provided by region_props using an OpenCV hack.
    
    FAST compared to region_props_connected (see `tests/results.txt`)

    This version is based on findContours.
    """
    # computes contours of all objects in image
    cnts, _ = cv2.findContours(bw_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # creates dictionaries of properties for each object
    objects = []

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
        # Deeper discussion of cv2.drawContours here:
        #https://learnopencv.com/contour-detection-using-opencv-python-c/
        mask = np.zeros(bw_frame.shape, np.uint8)
        cv2.drawContours(image=mask, contours=[cnt], contourIdx=0, 
                            color=255, thickness=-1)
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
              bw_frame, width_border)

        # adds dictionary for this object to list of objects in current frame
        objects += [obj]

    return objects


def region_props_skimage(bw_frame, n_frame=-1, width_border=5):
    """
    Extracts region properties using sci-kit image library instead
    of OpenCV. OpenCV does not provide easy access to many of the
    properties available with sci-kit image, such as the image in
    the bounding box and solidity (a measure related to convexity).
    """
    # labels frame
    frame_labeled = skimage.measure.label(bw_frame)
    # identifies the different objects in the frame
    region_props = skimage.measure.regionprops(frame_labeled)

    # creates dictionaries of properties for each object
    objects = []

    # loads properties for each object in frame
    for i, props in enumerate(region_props):
        # creates dictionary of bubble properties for one frame
        obj = {}
        
        # loads and stores region properties
        obj['centroid'] = props.centroid # as (row, col)
        obj['area'] = props.area
        obj['orientation'] = props.orientation
        obj['major axis'] = props.major_axis_length
        obj['minor axis'] = props.minor_axis_length
        obj['bbox'] = props.bbox # (row_min, col_min, row_max, col_max)
        # saves frame number
        if n_frame >= 0:
            obj['frame'] = n_frame
        obj['on border'] = is_on_border(props.bbox, frame_labeled, 
                                                    width_border)
        # solidity is the ratio of the area to the area of the convex hull
        obj['solidity'] = props.solidity
        # image of obj within bbox
        obj['image'] = props.image 
        # centroid relative to bounding box (bbox) as (row, col)
        obj['local centroid'] = props.local_centroid

        # stores object
        objects += [obj]

    return objects


def remove_small_objects(im, min_size):
    """
    Removes small objects in an image.
    Uses the findContours version because the tests suggest that it is faster.
    """
    return remove_small_objects_connected(im, min_size)


def remove_small_objects_find(im, min_size):
    """
    Removes small objects in image with OpenCV's findContours.
    Uses OpenCV to replicate `skimage.morphology.remove_small_objects`.

    Appears to be faster than with connectedComponentsWithStats (see
    run_opencv_tests.py).

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

    Appears to be slower than with findContours (see run_opencv_tests.py).

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

def track_obj(track_obj_method, track_kwargs, highlight_kwargs, assign_kwargs, ret_IDs=False):
    objects_archive = track_obj_method(track_kwargs, highlight_kwargs, assign_kwargs)

    # only returns IDs for each frame if requested
    # note: assumes objects_archive is aligned with start/end/every
    if ret_IDs:
        end = basic.get_frame_count(track_kwargs['vid_path'], track_kwargs['end'])
        frame_IDs = get_frame_IDs(objects_archive, track_kwargs['start'], end, track_kwargs['every'])
        return objects_archive, frame_IDs
    else:
        return objects_archive


def track_obj_cvvidproc(track_kwargs, highlight_kwargs, assign_kwargs):
    """
    Tracks objects using CvVidProc library.

    Parameters
    ----------
    track_kwargs : dictionary
        Variables related to object-tracking, indexed by variable name
    highlight_kwargs : dictionary
        Variables related to object-segmentation (highlighting), indexed by variable name
    assign_kwargs : dictionary
        Variables related to assigning object labels, indexed by variable name
    
    Returns
    -------
    objects_archive : dictionary
        Objects tracked in the video, indexed by ID #
    """
    # counts number of frames to analyze
    end = basic.get_frame_count(track_kwargs['vid_path'], track_kwargs['end'])
    n_frames = int( (end-track_kwargs['start'])/track_kwargs['every'] )

    # collects parameters for highlighting objects package (CvVidProc param)
    highlight_objects_pack = cvvidproc.HighlightObjectsPack(
        background=track_kwargs['bkgd'],
        struct_element=highlight_kwargs['selem'],
        threshold=highlight_kwargs['th'],
        threshold_lo=highlight_kwargs['th_lo'],
        threshold_hi=highlight_kwargs['th_hi'],
        min_size_hyst=highlight_kwargs['min_size_hyst'],
        min_size_threshold=highlight_kwargs['min_size_th'],
        width_border=highlight_kwargs['width_border'])

    # collects parameters for assigning objects package (CvVidProc param)
    # note that assigning objects is done purely in Python
    assign_objects_pack = cvvidproc.AssignObjectsPack(
        track_kwargs['assign_objects_method'],     # pass in function name for assignment as functor (not string)
        assign_kwargs)  # arguments for assign_objects_method

    # collects parameters for video management package (CvVidProc param)
    # fields not defined will be defaulted
    trackpack = cvvidproc.VidObjectTrackPack(
        vid_path=track_kwargs['vid_path'],
        highlight_objects_pack=highlight_objects_pack,
        assign_objects_pack=assign_objects_pack,
        frame_limit=n_frames, # must be int
        grayscale=True, # Whether to interpret the video has grayscale (TODO what's the difference b/w this and next param?)
        vid_is_grayscale=True, # Whether the video should be treated as already grayscale (optimization)
        crop_y=assign_kwargs['row_lo'],
        crop_height=assign_kwargs['row_hi']-assign_kwargs['row_lo'],
        print_timing_report=True)

    # starts timer
    print('tracking objects...')
    start_time = time.time()
    # tracks objects
    objects_archive = cvvidproc.TrackObjects(trackpack)
    # ends timer and prints results
    end_time = time.time()
    print('Tracked ({0:f} object(s); {1:f} s)'.format(len(objects_archive), end_time - start_time))

    return objects_archive


def track_obj_py(track_kwargs, highlight_kwargs, assign_kwargs):
    vid_path = track_kwargs['vid_path']
    highlight_method = track_kwargs['highlight_method']
    print_freq = track_kwargs['print_freq']
    start = track_kwargs['start']
    end = track_kwargs['end']
    every = track_kwargs['every']
    row_lo = assign_kwargs['row_lo']
    row_hi = assign_kwargs['row_hi']

    """
    ***TODO: install and implement decord VideoReader to speed up loading of
    frames: https://github.com/dmlc/decord***
    """
    # initializes ordered dictionary of object data from past frames and archive of all data
    objects_prev = OrderedDict()
    objects_archive = {}
    # initializes counter of current object label (0-indexed)
    next_ID = 0
    # chooses end frame to be last frame if given as -1
    end = basic.get_frame_count(vid_path, end)
    
    # loops through frames of video
    for f in range(start, end, every):

        # loads frame from video file
        frame, _ = basic.load_frame(vid_path, f)

        # crop the frame (only height is cropped in the current version)
        frame = frame[row_lo:row_hi, 0:frame.shape[1]]

        # extracts value channel of frame--including selem ruins segmentation
        val = basic.get_val_channel(frame)

        # highlights objects in the given frame
        objects_bw = highlight_method(val, track_kwargs['bkgd'], **highlight_kwargs)

        # finds objects and assigns IDs to track them, saving to archive
        next_ID = assign_objects(objects_bw, f, objects_prev, 
                                    objects_archive, next_ID, **assign_kwargs)

        if (f % print_freq*every) == 0:
            print('Processed frame {0:d} of range {1:d}:{2:d}:{3:d}.' \
                  .format(f, start, every, end))
        # a5 = time.time()
        # print('5 {0:f} ms.'.format(1000*(a5-a4)))

    return objects_archive
