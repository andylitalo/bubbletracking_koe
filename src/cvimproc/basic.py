"""
basic.py contains basic image-processing codes that are used by both improc.py
as vid.py

Author: Andy Ylitalo
Date: January 28, 2021
"""

import cv2
import scipy.ndimage
import numpy as np

import cvimproc.mask as mask
import genl.fn as fn

def adjust_brightness(im, brightness, sat=255):
    """
    Adjusts brightness of image by scaling all pixels by the
    `brightness` parameter.

    Parameters
    ----------
    im : (M, N, P) numpy array
        Image whose brightness is scaled. P >= 3,
        so RGB, BGR, RGBA acceptable.
    brightness : float
        Scaling factor for pixel values. If < 0, returns junk.
    sat : int
        Saturation value for pixels (usually 255 for uint8)

    Returns
    -------
    im : (M, N, P) numpy array
        Original image with pixel values scaled

    """
    # if image is 4-channel (e.g., RGBA) extracts first 3
    is_rgba = (len(im.shape) == 3) and (im.shape[2] == 4)
    if is_rgba:
        im_to_scale = im[:,:,:3]
    else:
        im_to_scale = im
    # scales pixel values in image
    im_to_scale = im_to_scale.astype(float)
    im_to_scale *= brightness
    # sets oversaturated values to saturation value
    im_to_scale[im_to_scale >= sat] = sat
    # loads result into original image
    if is_rgba:
        im[:,:,:3] = im_to_scale.astype('uint8')
    else:
        im = im_to_scale.astype('uint8')

    return im


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
        frame = scale_image(frame,scale)

    # Perform image filtering if requested
    if filterFrame:
        if removeBanner:
            ind = np.argmax(frame[:,0]>0)
            temp = frame[ind:,:]
            # temp = scipy.ndimage.gaussian_filter(temp, 0.03)
            temp = cv2.GaussianBlur(temp, (0,0), sigmaX=0.03, sigmaY=0.03)
            frame[ind:,:] = temp
        else:
            # frame = scipy.ndimage.gaussian_filter(frame, 0.03)
            frame = cv2.GaussianBlur(frame, (0,0), sigmaX=0.03, sigmaY=0.03)

    # Apply image transformation using homography matrix if passed
    if hMatrix is not None:
        temp = frame.shape
        frame = cv2.warpPerspective(frame,hMatrix,(temp[1],temp[0]))

    # Apply mask if needed
    if maskData is not None:
        frame = mask.mask_image(frame,maskData['mask'])
        if center:
            frame = rotate_image(frame,angle,center=maskData['diskCenter'],
                                     size=frame.shape)

    return frame


def get_val_channel(frame, selem=None):
    """
    Returns the value channel of the given frame.
    """
    # Convert reference frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Only interested in "value" channel to distinguish bubbles, filters result
    val = hsv[:,:,2] #skimage.filters.median(hsv[:,:,2], selem=selem).astype('uint8')

    return val


def load_frame(vid_path, num):
    """Loads frame from video using OpenCV and prepares for display in Bokeh."""
    cap = cv2.VideoCapture(vid_path)
    frame = read_frame(cap, num)

    return frame, cap


def read_frame(cap, num):
    """Reads frame given the video capture object."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)
    grabbed, frame = cap.read()

    return frame



def rotate_image(im,angle,center=[],crop=False,size=None):
    """
    Rotate the image about the center of the image or the user specified
    center. Rotate by the angle in degrees and scale as specified. The new
    image will be square with a side equal to twice the length from the center
    to the farthest.
    """
    temp = im.shape
    height = temp[0]
    width = temp[1]
    # Provide guess for center if none given (use midpoint of image window)
    if len(center) == 0:
        center = (width/2.0,height/2.0)
    if not size:
        tempx = max([height-center[1],center[1]])
        tempy = max([width-center[0],center[0]])
        # Calculate dimensions of resulting image to contain entire rotated image
        L = int(2.0*np.sqrt(tempx**2.0 + tempy**2.0))
        midX = L/2.0
        midY = L/2.0
        size = (L,L)
    else:
        midX = size[1]/2.0
        midY = size[0]/2.0

    # Calculation translation matrix so image is centered in output image
    dx = midX - center[0]
    dy = midY - center[1]
    M_translate = np.float32([[1,0,dx],[0,1,dy]])
    # Calculate rotation matrix
    M_rotate = cv2.getRotationMatrix2D((midX,midY),angle,1)
    # Translate and rotate image
    im = cv2.warpAffine(im,M_translate,(size[1],size[0]))
    im = cv2.warpAffine(im,M_rotate,(size[1],size[0]),flags=cv2.INTER_LINEAR)
    # Crop image
    if crop:
        (x,y) = np.where(im>0)
        im = im[min(x):max(x),min(y):max(y)]

    return im


def scale_image(im,scale):
    """
    Scale the image by multiplicative scale factor "scale".
    """
    temp = im.shape
    im = cv2.resize(im,(int(scale*temp[1]),int(scale*temp[0])))

    return im
