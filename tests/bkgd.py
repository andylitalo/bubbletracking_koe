"""test_cvvidproc.py

First, shows image of ISCO pump setup, closes when user clicks spacebar
Second, shows first frame of video of inner stream, closes when user clicks spacebar
Third, computes background of samples video. Should look like first frame^ w/o objects.
"""
import time
import cv2
import cvvidproc

import sys
sys.path.append('../src/')
import cvimproc.improc as improc


# test 0: shows image of isco pump
image = cv2.imread('../input/images/img.jpg')
cv2.imshow('Click spacebar', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# test 1: shows frame from vide of inner stream
vid_path = '../input/videos/vid2.mp4'

# loads video
cap = cv2.VideoCapture(vid_path)
ret, frame = cap.read()
cv2.imshow('Click spacebar', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

# test 2: computes background from video and shows result
bkgd = improc.compute_bkgd_med_thread(vid_path, num_frames=1000, max_threads=12)
cv2.imshow('Background -- click spacebar', bkgd)
cv2.waitKey(0)
cv2.destroyAllWindows()

# compare to previous, unoptimized, fully Python background algorithm in "Kornfield/ANALYSIS/improc-dev/bkg_alg_koe/test_python_speed.py"
