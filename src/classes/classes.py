# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:24:27 2020

@author: Andy
the Bubble class stores properties of a bubble from a video, both measured and
processed post factum.
"""
import numpy as np
import sys

# CONVERSIONS
um_2_m = 1E-6


class Bubble:
    def __init__(self, ID, fps, frame_dim, flow_dir, pix_per_um, props_raw=[],
                max_aspect_ratio=10):
        """
        frames: list of ints
        centroids: list of tuples of (row, col) of centroids of bubbles
        areas: same as centroids
        major axes: same as centroids
        minor axes: same as centroids
        orientation: same as centroids
        ID: int
        average flow direction: tuple of two floats normalized to 1
        average area: float
        average speed: float
        fps: float

        max_aspect_ratio : int
            Maximum aspect ratio of object to be classified as a bubble of the
            inner stream. Estimated based on seeing streaks with aspect ratio of
            12 and long bubbles having aspect ratio of ~5.

        true centroid is estimated location of centroid adjusted after taking
        account of the bubble being off screen/"on border."
        """
        # stores metadata
        self.metadata = {'ID':ID, 'fps':fps, 'frame_dim':frame_dim,
                         'flow_dir':flow_dir, 'pix_per_um':pix_per_um,
                         'max aspect ratio':max_aspect_ratio}
        # initializes storage of raw properties
        self.props_raw = {'frame':[], 'centroid':[], 'area':[], 'major axis':[],
                          'minor axis':[], 'orientation':[], 'bbox':[],
                          'on border':[]}
        # initializes storage of processed properties
        self.props_proc = {'speed':[],
                            'average area':None, 'average speed':None,
                           'average orientation':None, 'aspect ratio':[],
                           'average aspect ratio':None, 'true centroids':[],
                           'inner stream':-1, 'radius':[], 'average radius':None}
       # inner stream is 0 if bubble is likely in outer stream, 1 if
       # likely in inner stream, and -1 if unknown or not evaluated yet
        # loads raw properties if provided
        if len(props_raw) > 0:
            self.add_props(props_raw)

    ###### ACCESSORS ######
    def get_metadata(self, prop):
        return self.metadata[prop]

    def get_prop(self, prop, f):
        """Returns property at given frame f according to dictionary of frames."""
        try:
            prop = self.get_props(prop)[self.get_props('frame').index(f)]
            return prop
        except ValueError:
            print('Property {0:s} not available at frame {1:d}.'.format(prop, f))
            return None

    def get_props(self, prop):
        if prop in self.props_raw.keys():
            return self.props_raw[prop]
        elif prop in self.props_proc.keys():
            return self.props_proc[prop]
        else:
            print('Property not found in Bubble object. Returning None.')
            return None


    ###### MUTATORS ########
    def add_props(self, props):
        """props is properly organized dictionary. Only for raw properties."""
        for key in props.keys():
            if key in self.props_raw.keys():
                prop = props[key]
                # properties provided as lists are unchanged
                if isinstance(prop, list):
                    self.props_raw[key] += prop
                # properties provided as a number are converted to 1-elt list
                else:
                    self.props_raw[key] += [prop]
            else:
                print('Trying to add property not in Bubble.props_raw.')
        # keys for properties not provided in list
        keys_not_provided = (list(set(self.props_raw.keys()) - \
                                  set(props.keys())))
        for key in keys_not_provided:
            self.props_raw[key] += [None]

    def classify(self, v_inner):
        """
        Classifies bubble as inner or outer stream based on velocity cutoff.
        N.B.: If there is no inner stream (inner stream flow rate Q_i = 0),
        then the velocity v_inner will be computed to be nan, in which case this
        method will leave the classification as is.
        """
        # counts number of frames in which bubble appears
        n_frames = len(self.props_raw['frame'])
        # computes average speed [m/s] if not done so already
        if self.props_proc['average speed'] == None:
            self.proc_props()
        # classifies bubble as inner stream if velocity is greater than cutoff
        v = self.props_proc['average speed']

        # if the bubble's aspect ratio is too large, classify as error (-1)
        if np.max(self.props_proc['aspect ratio']) > self.metadata['max aspect ratio']:
            inner = -1
        # if no velocity recorded, classifies as inner stream (assumes too fast
        # to be part of outer stream)
        elif v == None:
            if n_frames == 1:
                inner = 1
            # unless there are more than 1 frames, in which case there is
            # probably an error
            else:
                inner = -1
        elif 0 < v and v < v_inner and n_frames > 1:
            inner = 0
        # if velocity is faster than lower limit for inner stream, classify as
        # inner stream
        elif (v >= v_inner) or (n_frames == 1):
            inner = 1
        # otherwise, default value of -1 is set to indicate unclear classification
        else:
            inner = -1

        self.props_proc['inner stream'] = inner

    def predict_centroid(self, f):
        """Predicts next centroid based on step sizes between previous centroids."""
        frames = self.props_raw['frame']
        centroids = self.props_raw['centroid']
        # no-op if centroid already provided for given frame
        if f in frames:
            # gets first index corresponding to requested frame
            i_frame = next((i for i in range(len(frames)) if frames[i] == f))
            return centroids[i_frame]

        # ensures at least 1 centroid
        assert len(centroids) > 0, \
                '{0:s} requires at least one centroid.' \
                .format(sys._getframe().f_code.co_name)
        # ensures same number of frames as centroids
        assert len(frames) == len(centroids), \
                '{0:s} requires equal # frames and centroids' \
                .format(sys._getframe().f_code.co_name)

        # if only 1 centroid provided, assume previous one was off screen in
        # direction opposite the flow direction
        if len(centroids) == 1:
            centroid = centroids[0]
            frame = frames[0]
            # estimates previous centroid assuming just offscreen
            centroid_prev = self.offscreen_centroid(centroid)
            # inserts previous centroid and frame
            centroids = [centroid_prev, centroid]
            frames = [frame-1, frame]

        # computes linear fit of previous centroids vs. frame
        # unzips rows and columns of centroids
        rows, cols = list(zip(*centroids))
        a_r, b_r = np.polyfit(frames, rows, 1)
        a_c, b_c = np.polyfit(frames, cols, 1)
        # predicts centroid for requested frame with linear fit
        centroid_pred = (a_r*f + b_r, a_c*f + b_c)

        return centroid_pred


    def proc_props(self):
        """Processes data to compute processed properties, mostly averages."""
        n_frames = len(self.props_raw['frame'])
        # computes average area [pix^2]
        area = self.props_raw['area']
        self.props_proc['average area'] = np.mean(area)
        # computes radius [um] as geometric mean of three diameters of the
        # bubble divided by 8 to get radius. Assumes symmetry about major axis
        # such that diameter in the other two dimensions is the minor axis
        self.props_proc['radius'] = [((maj*min*min)**(1.0/3)/8) / \
                                    self.metadata['pix_per_um'] for maj, min in\
                                    zip(self.props_raw['major axis'], \
                                    self.props_raw['minor axis'])]

        # computes aspect ratio and average (max prevents divide by 0)
        self.props_proc['aspect ratio'] = [self.props_raw['major axis'][i] /  \
                                        max(1, self.props_raw['minor axis'][i]) \
                                        for i in range(n_frames)]
        self.props_proc['average aspect ratio'] = np.mean( \
                                                self.props_proc['aspect ratio'])

        # computes average speed
        v_list = []
        fps = self.metadata['fps']
        dt = 1/fps # [s]
        flow_dir = self.metadata['flow_dir']
        pix_per_um = self.metadata['pix_per_um']
        frame_list = self.props_raw['frame']
        centroid_list = self.props_raw['centroid']
        for i in range(n_frames-1):
            diff = np.array(centroid_list[i+1]) - np.array(centroid_list[i])
            d = np.dot(diff, flow_dir)/pix_per_um*um_2_m # [m]
            t = dt*(frame_list[i+1]-frame_list[i]) # [s]
            v_list += [d/t] # [m/s]

        if len(v_list) > 0:
            # assumes speed in first frame is same as the second frame
            # to overcome loss of index in difference calculation of speed
            v_list.insert(0, v_list[0])
            self.props_proc['speed'] = v_list
            self.props_proc['average speed'] = np.mean(v_list)


    ### HELPER FUNCTIONS ###
    def offscreen_centroid(self, centroid):
        """
        Estimates previous centroid assuming just offscreen opposite flow
        direction.
        """
        # extracts centroid
        row, col = centroid
        # gets opposite direction from flow
        rev_dir = -np.array(self.metadata['flow_dir'])

        frame_dim = self.metadata['frame_dim']
        # computes steps to boundary in row and col directions
        n_r = self.steps_to_boundary(row, frame_dim[0], rev_dir[0])
        n_c = self.steps_to_boundary(col, frame_dim[1], rev_dir[1])

        # takes path requiring fewest steps
        if n_r <= n_c:
            row_off = row + n_r*rev_dir[0]
            col_off = col + n_r*rev_dir[1]
        else:
            row_off = row + n_c*rev_dir[0]
            col_off = col + n_c*rev_dir[1]

        return (row_off, col_off)

    def steps_to_boundary(self, x, x_max, s, x_min=0):
        """Computes number of steps s to boundary (x_min or x_max)."""
        assert x_max >= x and x_min <= x, 'x must be in range (x_min, x_max).'
        # pointing towards minimum boundary
        if s < 0:
            n = (x_min - x) / s
        elif s > 0:
            n = (x_max - x) / s
        else:
            n = float('inf')

        return n


########################## FILEVIDEOSTREAM CLASS ##############################

from threading import Thread
from queue import Queue
import cv2


class FileVideoStream:
    """
    Class for handling threaded loading of videos.
    Source:
    https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-
    videocapture-and-opencv/
    """
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
   				# if the `grabbed` boolean is `False`, then we have
   				# reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
   				# add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
   		# return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
   		# indicate that the thread should be stopped
        self.stopped = True
