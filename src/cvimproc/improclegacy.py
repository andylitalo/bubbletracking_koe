"""
improclegacy.py contains image-processing methods not currently in use.

I moved these methods here to reduce the clutter in `improc.py`. Eventually,
I may delete this file entirely.

***NOTE: THESE METHODS ARE NOT FUNCTIONAL BECAUSE THEY LACK PROPER IMPORTS***
Please move them to `improc.py` to use them.

Author: Andy Ylitalo
Date: July 1, 2021
"""


############################# METHOD DEFINITIONS ##############################


def highlight_obj_hyst(frame, bkgd, th_lo, th_hi, width_border, selem,
                          min_size, ret_all_steps=False):
    """
    Version of highlight_obj() that uses a hysteresis filter.
    """
    assert (len(frame.shape) == 2) and (len(bkgd.shape) == 2), \
        'improc.highlight_obj_hyst() only accepts 2D frames.'
    assert th_lo < th_hi, \
        'In improc.highlight_objects_hyst(), low threshold must be lower.'

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