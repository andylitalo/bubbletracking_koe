"""
main_helper.py contains helper functions for `src/main.py` to reduce
clutter and simplify modifications to the procedure.

Date: June 22, 2021
Author: Andy Ylitalo
"""

# imports standard libraries
import pickle as pkl
import os
import shutil
import glob

# imports custom libraries
# from libs
import genl.fn as fn
import cvimproc.improc as improc
import cvimproc.mask as mask
import cvimproc.ui as ui
import genl.flow as flow
import cvimproc.basic as basic
from classes.classes import Bubble, TrackedObject

# global conversions
import genl.conversions as conv
# configuration file
import config as cfg



def collect_kwargs(p, vid_path, bkgd, mask_data, flow_dir, row_lo, row_hi, 
                    v_max, v_interf, pix_per_um, remember_objects=False,
                    ellipse=True, ObjectClass=Bubble, object_kwargs={}):
    """
    Collects keyword arguments for CvVidProc's object-tracking algorithm.
    See API for CvVidProc library on UkoeHB's github for more details.

    Parameters
    ----------
    p : dictionary
        Contains parameters from input file (see `genl/readin.py`)
    vid_path : string
        Path to video where we will track objects
    bkgd : (M x N) numpy array of uint8s
        Background (computed with median, most likely) to subtract from frames
        of video for identifying objects in foreground
    mask_data : dictionary
        Contains data required for masking images of video
        See `mask.create_polygonal_mask_data()` for more info
    flow_dir : 2-tuple of floats
        Normalized vector indicating flow direction in (row, column) format
    row_lo, row_hi : int
        Lowest and highest row of image to keep after cropping
    v_max : float
        Predicted velocity at the center of the inner stream [m/s]
    pix_per_um : float
        Pixels per micron conversion factor
    remember_objects : bool, opt (default=False)
        If True, object-tracking will extrapolate position of objects that 
        are lost in tracking until they are extrapolated off screen or found
        again
        ***This option is still buggy so I'm leaving it False***
    ellipse : bool, optional
        If True, region_props will fit an ellipse to the object to compute
            major axis, minor axis, and orientation (default True)
    ObjectClass : class, opt (default=Bubble)
        Class of objects to track (TrackedObject is most general, but
        we will use Bubble since this analysis is specific to bubbles in 
        microfluidic flow)
    object_kwargs : dictionary
        Keyword arguments used in the initialization of an object from ObjectClass

    Returns
    -------
    track_kwargs, highlight_kwargs, assign_kwargs : dictionary
        Keyword arguments for object-tracking, segmentation, and labeling
    """
    # object-tracking keyword arguments
    track_kwargs = {'vid_path' : vid_path,
        'bkgd' : bkgd,
        'assign_objects_method' : cfg.assign_method,
        'highlight_method' : cfg.highlight_method, # only used for pure-Python analysis
        'start' : p['start'],
        'end' : p['end'],
        'every' : p['every']
    }

    # object highlighting/segmentation keyword arguments
    highlight_kwargs = {'th' : p['th'],
        'th_lo' : p['th_lo'],
        'th_hi' : p['th_hi'],
        'min_size_hyst' : p['min_size_hyst'],
        'min_size_th' : p['min_size_th'],
        'width_border' : p['width_border'],
        'selem' : p['selem'],
        'mask_data' : mask_data
    }

    # additional variables required for method to measure distance between objects
    m_2_pix = conv.m_2_um*pix_per_um # conversion factors for meters to pixels
    d_fn_kwargs = {
        'axis' : flow_dir,
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'v_max' : v_max*m_2_pix,  # convertd max velocity from [m/s] to [pix/s]
        'v_interf' : v_interf*m_2_pix, # convert interfacial velocity from [m/s] to [pix/s]
        'fps' : fn.parse_vid_path(vid_path)['fps'],
    }
    # label assignment/tracking keyword arguments
    assign_kwargs = {
        'fps' : fn.parse_vid_path(vid_path)['fps'],  # extracts fps from video filepath
        'd_fn' : cfg.d_fn,
        'd_fn_kwargs' : d_fn_kwargs,
        'width_border' : p['width_border'],
        'min_size_reg' : p['min_size_reg'],
        'row_lo' : row_lo,
        'row_hi' : row_hi,
        'remember_objects' : remember_objects,
        'ellipse' : ellipse,
        'ObjectClass' : ObjectClass,
        'object_kwargs' : object_kwargs,
    }               

    return track_kwargs, highlight_kwargs, assign_kwargs


def get_bkgd(vid_path, data_dir, row_lo, row_hi,
            num_frames_for_bkgd, use_prev_bkgd):
    """
    Gets background for video. If requested and available, existing
    background can be used instead of computing from scratch.

    Parameters
    ----------
    vid_path : string
        Filepath to video requiring a background frame
    data_dir : string
        Path to directory containing data from previous analyses of this video
    row_lo, row_hi : int
        Lowest and highest row of image to keep after cropping
    num_frames_for_bkgd : int
        Number of the initial frames of the video to use for computing the 
        background. `readin.py` ensures that this is fewer than the total 
        number of frames in the video.
    use_prev_bkgd : bool
        If True, will use previously computed background if available

    Returns
    -------
    bkgd : (M x N) numpy array of uint8s
        Background computed by taking the median of each pixel value across
        several frames.
    """
    # gets background for image subtraction later on
    data_existing = glob.glob(os.path.join(data_dir, 'f_*.pkl'))
    if len(data_existing) > 0 and use_prev_bkgd:
        # loads background from first existing data file
        with open(data_existing[0], 'rb') as f:
            data_prev = pkl.load(f)
        bkgd = data_prev['metadata']['bkgd']
    else:
        # computes background with median filtering
        bkgd = improc.compute_bkgd_med_thread(
            vid_path,
            vid_is_grayscale=True,  #assumes video is already grayscale
            num_frames=num_frames_for_bkgd,
            crop_y=row_lo,
            crop_height=row_hi-row_lo)

    return bkgd


def get_flow_props(vid_path, p):
    """
    Gets properties of the microfluidic flow.

    Parameters
    ----------
    vid_path : string
        Path to video of microfluidic flow
    p : dictionary
        Contains parameters from input file (see `genl/readin.py`)

    Returns
    -------
    dp : float
        Predicted pressure drop across observation capillary (negative) [Pa]
    R_i : float
        Predicted inner stream radius [m]
    v_max, v_interf : float
        Predicted velocities at the center and at the interface of the 
        inner stream [m/s]
    Q_i, Q_o : float
        Inner and outer stream flow rates [m^3/s]
    pix_per_um : float
        Number of pixels per micron
    """
    # extracts parameters recorded in video name
    vid_params = fn.parse_vid_path(vid_path)
    Q_i = conv.uLmin_2_m3s*vid_params['Q_i'] # inner stream flow rate [m^3/s]
    Q_o = conv.uLmin_2_m3s*vid_params['Q_o'] # outer stream flow rate [m^3/s]
    d = conv.mm_2_m*vid_params['d']
    mag = vid_params['mag'] # magnification of objective (used for pix->um conv)
    # gets unit conversion based on camera and microscope magnification
    pix_per_um = conv.pix_per_um[p['camera']][mag]

    # computes pressure drop [Pa], inner stream radius [m], and max velocity
    #  [m/s] for Poiseuille sheath flow
    dp, R_i, v_max = flow.get_dp_R_i_v_max(p['eta_i'], p['eta_o'], p['L'],
                                        Q_i, Q_o, p['R_o'], SI=True)
    # computes velocity at interface of inner stream [m/s]
    v_interf = flow.v_interf(Q_i, Q_o, p['eta_i'], p['eta_o'], p['R_o'], p['L'])

    return dp, R_i, v_max, v_interf, Q_i, Q_o, pix_per_um, d


def get_mask(vid_path, check=True):
    """
    Has user create mask (if not already present). Along with mask, returns
    direction of the fluid flow and the upper and lower rows to crop the
    image with (helpful for reducing memory requirements for loading images).

    Parameters
    ----------
    vid_path : string
        Filepath to video requiring a mask
    check : bool, opt (default=True)
        If True and a mask already exists for the video, asks user to check
        the quality of the mask
    
    Returns
    -------
    mask_data : dictionary
        Contains data required for masking images of video
        See `mask.create_polygonal_mask_data()` for more info
    flow_dir : 2-tuple of floats
        Normalized vector indicating flow direction in (row, column) format
    row_lo, row_hi : int
        Lowest and highest row of image to keep after cropping
    """
    # loads mask data; user creates new mask by clicking if none available
    first_frame, _ = basic.load_frame(vid_path, 0)
    flow_dir, mask_data = ui.click_flow(first_frame,
                                    vid_path[:-4]+'_mask.pkl', check=check)
    # computes minimum and maximum rows for object tracking computation
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)

    return mask_data, flow_dir, row_lo, row_hi 


def get_paths(p, replace):
    """
    Gets directories and paths to data.

    Parameters
    ----------
    p : dictionary
        Contains parameters from input file (see `genl/readin.py`)
    replace : bool
        If True, replaces/overwrites existing data if present

    Returns
    -------
    vid_path, data_path : string
        Paths to video to analyze and data file generated from analysis
    vid_dir, data_dir, figs_dir : string
        Paths to directories containing video file, data file, and figures
    stop : bool
        If True, something indicated that the analysis should stop
    """
    # initially assumes analysis should not be stopped
    stop = False

    # defines filepath to video
    vid_path = os.path.join(cfg.input_dir, p['vid_subdir'], p['vid_name'])
    
    # checks that video has the requested frames
    # subtracts 1 since "end" gives an exclusive upper bound [start, end)
    if not basic.check_frames(vid_path, p['end']-1):
        print('Terminating analysis. Please enter valid frame range next time.')
        stop = True

    # defines directory to video data and figures
    vid_dir = os.path.join(p['vid_subdir'], p['vid_name'][:-4], p['input_name'])
    data_dir = os.path.join(cfg.output_dir, vid_dir, cfg.data_subdir)
    figs_dir = os.path.join(cfg.output_dir, vid_dir, cfg.figs_subdir)

    # creates directories recursively if they do not exist
    fn.makedirs_safe(data_dir)
    fn.makedirs_safe(figs_dir)

    # defines name of data file to save
    data_path = os.path.join(data_dir,
                            'f_{0:d}_{1:d}_{2:d}.pkl'.format(p['start'],
                                                        p['every'], p['end']))
    # if the data file already exists in the given directory, end analysis
    # upon request
    # if replacement requested, previous background may be used upon request
    if os.path.isfile(data_path):
        if not replace:
            print('{0:s} already exists. Terminating analysis.'.format(data_path))
            stop = True

    return vid_path, data_path, vid_dir, data_dir, figs_dir, stop


def save_data(objs, frame_IDs, p, track_kwargs, highlight_kwargs, assign_kwargs, 
                vid_path, input_path, data_path):
    """
    Saves data and metadata from object-tracking.

    Parameters
    ----------
    objs : dictionary
        Objects tracked by algorithm indexed by ID #
    frame_IDs : dictionary
        Indexed by frame number, lists ID #s of objects in each frame
    p : dictionary
        Contains parameters from input file (see `genl/readin.py`)
    track_kwargs, highlight_kwargs, assign_kwargs : dictionary
        Keyword arguments for object-tracking, segmentation, and labeling.
        See CvVidProc API for more details.
    vid_path, input_path, data_path : string
        Filepaths to video analyzed, input file, and data saved

    Returns nothing.
    """
    # collects metadata -- especially adds highlight_kwargs for quick highlight call
    metadata = {'frame_IDs' : frame_IDs, 'highlight_kwargs' : highlight_kwargs}
    param_dicts = (p, track_kwargs, assign_kwargs)
    for d in param_dicts:
        for key in d:
            metadata[key] = d[key]
        
    # stores data
    data = {}
    data['objects'] = objs
    data['frame IDs'] = frame_IDs
    data['metadata'] = metadata

    # splits filepaths to get data directory and input filename
    data_dir, _ = os.path.split(data_path)
    _, input_filename = os.path.split(input_path)
    # also saves copy of input file and mask file
    shutil.copyfile(input_path, os.path.join(data_dir, input_filename))
    shutil.copyfile(vid_path[:-4] + '_mask.pkl',
                        os.path.join(data_dir, 'mask.pkl'))

    # saves data
    with open(data_path, 'wb') as f:
        pkl.dump(data, f)

    return 