# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:03:01 2020

@author: Andy
vid.py contains functions useful for manipulating videos.
"""

# directs system to source directory
import sys
sys.path.append('../')

import numpy as np
import cv2
import scipy.ndimage
from bokeh.io import show, push_notebook
import time

# imports custom libraries
import genl.fn as fn
import cvimproc.improc as improc
import mask
import plot.improc as pltim



def check_frames(vid_path, n):
    """
    Checks if frame number n is within range of frames in the video at vid_path.

    Parameters
    ----------
    vid_path : string
        Filepath to video
    n : int
        Frame number to check (0-indexed)

    Returns
    -------
    contains_frame : bool
        True if video has at least n frames; False otherwise
    """
    n_frames = count_frames(vid_path)
    contains_frame = n < n_frames
    if not contains_frame:
        print('{0:d}th frame requested, but only {1:d} frames available.' \
                .format(n, n_frames))

    return contains_frame


def count_frames(path, override=False):
    """
    This method comes from https://www.pyimagesearch.com/2017/01/09/
    count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    written by Adrian Rosebrock.
    The method counts the number of frames in a video using cv2 and
    is robust to the errors that may be encountered based on what
    dependencies the user has installed.

    Parameters
    ----------
    path : string
        Direction to file of video whose frames we want to count
    override : bool (default = False)
        Uses slower, manual counting if set to True

    Returns
    -------
    n_frames : int
        Number of frames in the video. -1 is passed if fails completely.
    """
    video = cv2.VideoCapture(path)
    n_frames = 0
    if override:
        n_frames = count_frames_manual(video)
    else:
        try:
            if fn.is_cv3():
                n_frames = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
            else:
                n_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            n_frames = count_frames_manual(video)

    # release the video file pointer
    video.release()

    return n_frames


def count_frames_manual(video):
    """
    This method comes from https://www.pyimagesearch.com/2017/01/09/
    count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    written by Adrian Rosebrock.
    Counts frames in video by looping through each frame.
    Much slower than reading the codec, but also more reliable.
    """
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()
        # check if we reached end of video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total


def extract_frame(Vid,nFrame,hMatrix=None,maskData=None,filterFrame=False,
                  removeBanner=True,center=True,scale=1,angle=0):
    """
    Extracts nFrame'th frame and scales by 'scale' factor from video 'Vid'.
    """
    Vid.set(1,nFrame)
    ret, frame = Vid.read()
    if not ret:
        print('Frame not read')
    else:
        frame = frame[:,:,0]

    # Scale the size if requested
    if scale != 1:
        frame = improc.scale_image(frame,scale)

    # Perform image filtering if requested
    if filterFrame:
        if removeBanner:
            ind = np.argmax(frame[:,0]>0)
            temp = frame[ind:,:]
            temp = scipy.ndimage.gaussian_filter(temp, 0.03)
            frame[ind:,:] = temp
        else:
            frame = scipy.ndimage.gaussian_filter(frame, 0.03)

    # Apply image transformation using homography matrix if passed
    if hMatrix is not None:
        temp = frame.shape
        frame = cv2.warpPerspective(frame,hMatrix,(temp[1],temp[0]))

    # Apply mask if needed
    if maskData is not None:
        frame = mask.mask_image(frame,maskData['mask'])
        if center:
            frame = improc.rotate_image(frame,angle,center=maskData['diskCenter'],
                                     size=frame.shape)

    return frame


def load_frame(vid_path, num, vert_flip=True, bokeh=True):
    """Loads frame from video using OpenCV and prepares for display in Bokeh."""
    # assert num < count_frames(vid_path), 'Frame number in vid.load_frame() must be less than total frames.'
    cap = cv2.VideoCapture(vid_path)
    frame = read_frame(cap, num)
    if bokeh:
        frame = pltim.bokehfy(frame)

    return frame, cap


def read_frame(cap, num):
    """Reads frame given the video capture object."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)
    grabbed, frame = cap.read()

    return frame


def show_frame(vid_path, start_frame, pix_per_um, vert_flip=True,
               fig_size_red=0.5, brightness=1.0, show_fig=True):
    """
    vert_flip:  # code for flipping vertically with cv2.flip()
    # dw is the label on the axis
    # x_range is the extent shown (so it should match dw)
    # width is the width of the figure box
    """
    frame, cap = load_frame(vid_path, start_frame)
    p, im = pltim.format_frame(frame, pix_per_um, fig_size_red, brightness=brightness)
    if show_fig:
        show(p, notebook_handle=True)

    return p, im, cap


def view_video(vid_path, start_frame, pix_per_um, time_sleep=0.3,
               brightness=1.0, vert_flip=True, show_frame_num=False,
               fig_size_red=0.5):
    """
    vert_flip:  # code for flipping vertically with cv2.flip()
    Functions and setup were adapted from:
    https://stackoverflow.com/questions/27882255/is-it-possible-to-display-an-opencv-video-inside-the-ipython-jupyter-notebook
    """
    p, im, cap = show_frame(vid_path, start_frame, pix_per_um,
                            brightness=brightness, fig_size_red=fig_size_red)
    if show_frame_num:
        f = start_frame
    while True:
        ret, frame = cap.read()
        # displays frame number on lower-left of screen
        if show_frame_num:
            f += 1
            white = (255, 255, 255)
            h = frame.shape[0]
            frame = cv2.putText(img=frame, text=str(f), org=(10, h-10),
                                    fontFace=0, fontScale=2, color=white,
                                    thickness=3)
        # formats frame for viewing
        frame = improc.adjust_brightness(pltim.bokehfy(
                frame, vert_flip=vert_flip), brightness)
        # displays frame
        im.data_source.data['image']=[frame]
        push_notebook()
        # waits before showing the next frame
        time.sleep(time_sleep)
