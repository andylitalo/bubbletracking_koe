# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:03:01 2020

@author: Andy
vid.py contains functions useful for manipulating videos.
"""

import numpy as np
import cv2
from bokeh.io import show, push_notebook
import time

# imports custom libraries
import genl.fn as fn
import cvimproc.basic as basic
import cvimproc.mask as mask
import cvimproc.pltim as pltim




def show_frame(vid_path, start_frame, pix_per_um, vert_flip=True,
               fig_size_red=0.5, brightness=1.0, show_fig=True):
    """
    vert_flip:  # code for flipping vertically with cv2.flip()
    # dw is the label on the axis
    # x_range is the extent shown (so it should match dw)
    # width is the width of the figure box
    """
    frame, cap = basic.load_frame(vid_path, start_frame)
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
        frame = basic.adjust_brightness(pltim.bokehfy(
                frame, vert_flip=vert_flip), brightness)
        # displays frame
        im.data_source.data['image']=[frame]
        push_notebook()
        # waits before showing the next frame
        time.sleep(time_sleep)
