"""
basic.py contains basic image-processing codes that are used by improc.py,
mask.py, and vid.py, and other libraries that call improc.py. By storing them in
this separate library, the libraries will not call each other and confuse
Python.

Author: Andy Ylitalo
Date: January 28, 2021
"""

import cv2
import numpy as np

# imports custom libraries
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
                n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                n_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            print('OpenCV cannot access frame count--counting frames manually')
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


def cvify(im):
    """
    Makes image suitable for OpenCV functions, i.e., converts to 0-255 scale
    uint8 array.

    Parameters
    ----------
    im : numpy array
        Image to make suitable for OpenCV. Can be any type.

    Returns
    -------
    im_cv : numpy array of uint8s
        Image suitable for OpenCV algorithms (0-255 scale, uint8)
    """
    # converts boolean to 0-255 scale, uint8
    if im.dtype == 'bool':
        im = im.astype('uint8', copy=False)
        im *= 255
    # converts float array scaled from 0-1 to uint8, 0-255 scale
    elif im.dtype == 'float' and np.max(im) <= 1:
        im *= 255
        im = im.astype('uint8', copy=False)
    elif str(im.dtype).find('int') >= 0:
        im = im.astype('uint8', copy=False)
    elif im.dtype != 'uint8':
        print('basic.cvify(): image type {0:s} not recognized'.format(
                                                                str(im.dtype)))

    return im


def fill_holes(im_bw):
    """
    Fills holes in image solely using OpenCV to replace
    `fill_holes` for porting to C++.

    Based on:
     https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    Parameters
    ----------
    im_bw : numpy array of uint8
        Image whose holes are to be filled. 0s and 255s

    Returns
    -------
    im : numpy array of uint8
        Image with holes filled, including those cut off at border. 0s and 255s
    """
    # formats image for OpenCV (and copies it)
    im_bw = cvify(im_bw)
    im_floodfill = im_bw.copy()
    # finds point connected to background
    for x_bkgd in range(im_bw.shape[1]):
        if im_floodfill[0, x_bkgd] == 0:
            break
        elif x_bkgd == im_bw.shape[1]-1:
            print('could not find 0 value in basic.fill_holes().')
    # fills bkgd with white (assuming origin is contiguously connected with bkgd)
    # seed point is in the format (x,y)
    seed_pt = (x_bkgd, 0)
    h, w = im_bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, None, seed_pt, 255)
    # inverts image (black -> white and white -> black)
    im_inv = cv2.bitwise_not(im_floodfill)
    # combines inverted image with original image to fill holes
    im_filled = im_bw | im_inv

    return im_filled


def get_frame_count(vid_path, end):
    """
    Gets the index of the final frame requested. This is just
    `end` unless `end`==-1, in which case it is the number of 
    frames in the video (we do not subtract one since `end` is 
    treated as an exclusive upper bound).

    Provides flexibility by allowing user to specify final frame or request it
    to be counted for them (by setting end = -1).

    Parameters
    ----------
    vid_path : string
        Path to video whose frames we want to count
    end : int
        Value given for the final frame to count (could be -1 for last frame)
    
    Returns
    -------
    final_frame_num : int
        index of final frame (`= end` unless `end == -1`, in which 
        case it's the number of frames in the video)
    """
    if end == -1:
        final_frame_num = count_frames(vid_path)
    else:
        final_frame_num = end

    return final_frame_num
    

def get_val_channel(frame, selem=None):
    """
    Returns the value channel of the given frame.
    """
    # Convert reference frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Only interested in "value" channel to distinguish objects, filters result
    val = hsv[:,:,2] #skimage.filters.median(hsv[:,:,2], selem=selem).astype('uint8')

    return val


def is_color(im):
    """Returns True if the image is a color image (3 channels) and false if not."""
    return len(im.shape) == 3

    
def load_frame(vid_path, num):
    """Loads frame from video using OpenCV and prepares for display in Bokeh."""
    cap = cv2.VideoCapture(vid_path)
    frame = read_frame(cap, num)

    return frame, cap


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


def scale_image(im,scale):
    """
    Scale the image by multiplicative scale factor "scale".
    """
    temp = im.shape
    im = cv2.resize(im,(int(scale*temp[1]),int(scale*temp[0])))

    return im
