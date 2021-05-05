# bubbletracking_koe

Self-contained bubble-tracking code only using OpenCV for image-processing functions.
Using exclusively OpenCV will allow for porting into C++.

## Dependencies

OpenCV
Numpy
tkinter
argparse
pickle
ctypes
matplotlib
pandas 

## Running Bubble Tracking

## Post-processing Analysis and Validation

Often, the best validation of image processing is visual. The `analysis/highlight.py`
method 

## TODOs

Apply homographic transformation so inner stream is horizontal and mask cuts off all of the outer stream
(currently, the mask is applied as a bounding box)
Actually show result of masking when asking user if mask is okay
Remove second request for mask (can just ask user to mask region of interest and ensure that first click gives flow
direction)

