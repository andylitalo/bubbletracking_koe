"""
classes.py contains classes for tracking objects:

- The TrackedObject class stores properties of an object that was identified and
tracked in a video and can process some of those properties to derive new ones.
- The Bubble class is a sub-class of the TrackedObj class and includes some 
methods specific to the flow of bubbles down a stream.

Author: Andy Ylitalo
Date: May 11, 2020
"""

# standard libraries
import numpy as np
import sys

# conversions
import genl.conversions as conv

    
class TrackedObject:
    """An object for storing information about tracked objects in a video.

    The object stores metadata for the video, properties of the object
    measured after segmentation, and properties that are computed or 
    processed using methods belonging to this object.

    ...

    Attributes
    ----------
    metadata : dictionary
        Metadata of video. Keys are ID, fps, frame_dim (see "Parameters")
    props_raw : dictionary
        Items are:
        - frame : list of frame numbers in which object was detected
        - centroid : list of 2-tuples of centroid coordinates (row, col)
        - area : list of the number of pixels contained in object
        - major axis : list of the lengths of the major axis of the object
            in each frame (= 2 * semimajor axis)
        - minor axis : list of the lengths of the minor axis of the object
            in each frame (= 2 * semiminor axis)
        - orientation : list of the angles of orientation of major axes 
            from vertical axis [rad]
            (source: https://datascience.stackexchange.com/questions/
            85064/where-is-the-rotated-angle-actually-located-in-fitellipse-method)
        - bbox : list of the bounding boxes of object in each frame
            (row_min, col_min, row_max, col_max)
        - on border : list of booleans indicating if object is on 
            border (True) or not (False)
    props_proc : dictionary
        Items are:
        - speed : speed of object calculated with backward difference [pix/s]
        - average <prop> : average of <prop> property
        - true centroids : same as centroid property, but removes centroids that
            were extrapolated (see `predict_centroid` method of Bubble object) and
            takes account of the object being off screen/"on border."
            NOT IMPLEMENTED
    """

    def __init__(self, ID, fps, frame_dim, props_raw=[], **metadata):
        """
        Initializes an object tracked across multiple frames in a video.
       
        Parameters
        ----------
        ID : int
            Identification number of object
        fps : int
            frame rate of video [frames per second]
        frame_dim : 2-tuple or 3-tuple
            dimensions of the frames of the video (row, col[, channels])
        props_raw : list, opt (default=[])
            Raw, unprocessed properties of the object. These can be added
            during initialization or using a set method later.
        metadata : dictionary
            properties to store in metadata of object (useful for subclasses)
        """

        # stores metadata
        self.metadata = {'ID' : ID, 'fps' : fps, 'frame_dim' : frame_dim}
        for key in metadata.keys():
            self.metadata[key] = metadata[key]

        # initializes storage of raw properties
        self.props_raw = {'frame':[], 'centroid':[], 'area':[], 'major axis':[],
                          'minor axis':[], 'orientation':[], 'bbox':[],
                          'on border':[]}
        # initializes storage of processed properties
        self.props_proc = {'speed':[],
                            'average area':None, 'average speed':None,
                           'average orientation':None, 'aspect ratio':[],
                           'average aspect ratio':None, 'true centroids':[]
                          }
        # loads raw properties if provided
        if len(props_raw) > 0:
            self.add_props(props_raw)


    ###### ACCESSORS ######

    def get_all_props(self):
        """Returns all raw and processed properties of the object."""
        props = {}
        for prop_key in self.props_raw:
            props[prop_key] = self.props_raw[prop_key]
        for prop_key in self.props_proc:
            props[prop_key] = self.props_proc[prop_key]
           
        return props
            
    def get_metadata(self, prop):
        """Returns property `prop` (string) from metadata."""
        return self.metadata[prop]

    def get_prop(self, prop, f):
        """Returns property `prop` at given frame `f`."""
        # tries access property at desired frame
        try:
            prop = self.get_props(prop)[self.get_props('frame').index(f)]
            return prop
        # otherwise, no property returned
        except ValueError:
            print('Property {0:s} not available at frame {1:d}.'.format(prop, f))
            return None

    def get_props(self, prop):
        """Returns property `prop` for all frames."""
        # checks raw properties
        if prop in self.props_raw.keys():
            return self.props_raw[prop]
        # checks processed properties
        elif prop in self.props_proc.keys():
            return self.props_proc[prop]
        # otherwise, not found
        else:
            print('Property not found in object. Returning None.')
            return None

    ###### MUTATORS ########

    def add_props(self, props):
        """Adds properties from dictionary `props` to `props_raw`."""
        for key in props.keys():
            # checks for valid property
            if key in self.props_raw.keys():
                prop = props[key]
                # properties provided as lists are unchanged
                if isinstance(prop, list):
                    self.props_raw[key] += prop
                # properties provided as a number are converted to singlet list
                else:
                    self.props_raw[key] += [prop]
            else:
                print('Trying to add property not in props_raw.')


    def proc_props(self):
        """Computes processed properties, such as speed and average values."""

        # counts the number of frames
        n_frames = len(self.props_raw['frame'])

        # computes average area [pix^2]
        area = self.props_raw['area']
        self.props_proc['average area'] = np.mean(area)

        # uses width and height if major and minor axes were not computed
        if (len(self.props_raw['major axis']) == 0) or \
            (len(self.props_raw['minor axis']) == 0):
            major_list = [col_max - col_min for _, col_min, _, col_max in self.props_raw['bbox']]
            minor_list = [row_max - row_min for row_min, _, row_max, _ in self.props_raw['bbox']]
        else:
            major_list = self.props_raw['major axis']
            minor_list = self.props_raw['minor axis']
        # computes aspect ratio and average (max prevents divide by 0)
        self.props_proc['aspect ratio'] = [major_list[i] / \
                                            max(1, minor_list[i]) \
                                            for i in range(n_frames)]
        self.props_proc['average aspect ratio'] = np.mean( \
                                            self.props_proc['aspect ratio'])

        # computes average speed
        v_list = []
        fps = self.metadata['fps'] # [frames / s]
        dt = 1/fps # [s]
        frame_list = self.props_raw['frame']
        centroid_list = self.props_raw['centroid']
        # estimates speed with first-order difference
        for i in range(n_frames-1):
            # distance between consecutive centroids [pixels]
            # TODO account for shift in centroid if object is "on border"
            d = np.linalg.norm(np.array(centroid_list[i+1]) - np.array(centroid_list[i]))
            # time between consecutive frames
            t = dt*(frame_list[i+1]-frame_list[i]) # [s]
            v_list += [d/t] # [pixel/s]
        # shifts speeds by one index (so calculation becomes backward difference)
        if len(v_list) > 0:
            # assumes speed in first frame is same as the second frame so speed list
            # has as many elements as the other properties
            v_list.insert(0, v_list[0])
            self.props_proc['speed'] = v_list
            self.props_proc['average speed'] = np.mean(v_list)


    #### HELPER FUNCTIONS ####

    def predict_centroid(self, f):
        """
        Predicts centroid in frame `f` based on step sizes between previous 
        centroids.
        
        Parameters
        ----------
        f : int
            Frame number in which to predict centroid
        
        Returns
        -------
        centroid_pred : 2-tuple of ints
            Predicted centroid (row, column)
        """
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
                .format(sys._getframe().f_code.co_name) # grabs method name

        # if only 1 centroid provided, predicts the same centroid
        # *Sub-classes may incorporate object-specific properties for prediction
        if len(centroids) == 1:
            centroid_pred = centroids[0]
        else:
            # computes linear fit of previous centroids vs. frame
            rows, cols = list(zip(*centroids))
            a_r, b_r = np.polyfit(frames, rows, 1)
            a_c, b_c = np.polyfit(frames, cols, 1)
            centroid_pred = (a_r*f + b_r, a_c*f + b_c)

        return centroid_pred



class Bubble(TrackedObject):
    """
    A Bubble object is a TrackedObject with some additional properties.
    These properties are reserved for Bubble objects because we know
    a few things about a bubble flowing down a microfluidic channel
    that might not always be true for any object in a video, such as
    its direction of motion, 3D shape (ellipsoidal, roughly), and the
    conversion from pixels to physical length.
    """

    def __init__(self, ID, fps, frame_dim, props_raw=[], **metadata):
        """
        Same as for TrackedObject but adds some more properties to compute:
        - radius : list of floats
            computed assuming axisymmetry about the flow direction
        - average radius : float
            average of the list of radii
        - speed [m/s] : list of floats
            Speed converted from pix/s to m/s using pix_per_um conversion
        - average speed [m/s] : float
            average of speed [m/s]
        - inner stream : int
            -1 --> error, 0 --> bubble is in the outer stream, 1 --> bubble
            is in the inner stream of microfluidic sheath flow
        """

        # uses TrackedObject init fn to lay foundation for the object
        super().__init__(ID, fps, frame_dim, props_raw=props_raw, **metadata)
        # adds Bubble-specific properties
        self.props_proc['radius'] = []
        self.props_proc['average radius'] = None
        self.props_proc['speed [m/s]'] = []
        self.props_proc['average speed [m/s]'] = None
        # inner stream is 0 if bubble is likely in outer stream, 1 if
        # likely in inner stream, and -1 if unknown or not evaluated yet
        self.props_proc['inner stream'] = -1

       
    ###### MUTATORS ########
    def classify(self, v_interf, max_aspect_ratio=10):
        """
        Classifies bubble as inner or outer stream based on velocity cutoff.
        N.B.: If there is no inner stream (inner stream flow rate Q_i = 0),
        then the velocity v_interf will be computed to be nan, in which case this
        method will leave the classification as is.

        Parameters
        ----------
        v_interf : float
            Predicted speed of the sheath flow at the interface between the
            inner stream and the outer stream [m/s]
        max_aspect_ratio : float, optional
            Maximum aspect ratio to classify an object as a bubble (otherwise
            sets inner stream --> -1 to signify an error)

        Returns nothing.
        """
        # counts number of frames in which bubble appears
        n_frames = len(self.props_raw['frame'])
        # computes average speed [m/s] if not done so already
        if self.props_proc['average speed [m/s]'] == None:
            self.proc_props()
        # classifies bubble as inner stream if velocity is greater than cutoff
        v = self.props_proc['average speed [m/s]']

        # if the bubble's aspect ratio is too large, classify as error (-1)
        if np.max(self.props_proc['aspect ratio']) >= max_aspect_ratio:
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
        elif 0 < v and v < v_interf and n_frames > 1:
            inner = 0
        # if velocity is faster than lower limit for inner stream, classify as
        # inner stream
        elif (v >= v_interf) or (n_frames == 1):
            inner = 1
        # otherwise, default value of -1 is set to indicate unclear classification
        else:
            inner = -1

        self.props_proc['inner stream'] = inner


    def predict_centroid(self, f):
        """
        Predicts centroid in frame `f` based on previous centroids.
        Identical to TrackedObject.predict_centroid except if the object
        has only been detected in one frame, in which case the method assumes
        that the bubble was just off screen in the reverse flow direction in 
        the previous frame.
        
        Parameters
        ----------
        f : int
            Frame for which to predict the centroid of the bubble
        
        Returns
        -------
        centroid_pred : 2-tuple of floats
            Predicted centroid of object in (row, column) format
        """
        # loads frames and centroids recorded so far
        frames = self.props_raw['frame']
        centroids = self.props_raw['centroid']

        # same behavior if multiple or no centroids have been recorded
        if len(centroids) != 1:
            return super().predict_centroid(f)
        # otherwise, predicts centroid using knowledge of flow direction
        else:
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
        # computes TrackedObject processed properties in the same way
        super().proc_props()

        # computes radius [um] as geometric mean of three diameters of the
        # bubble divided by 8 to get radius. Assumes symmetry about major axis
        # such that diameter in the other two dimensions is the minor axis
        # uses width and height if major and minor axes were not computed
        if (len(self.props_raw['major axis']) == 0) or \
            (len(self.props_raw['minor axis']) == 0):
            major_list = [col_max - col_min for _, col_min, _, col_max in self.props_raw['bbox']]
            minor_list = [row_max - row_min for row_min, _, row_max, _ in self.props_raw['bbox']]
        else:
            major_list = self.props_raw['major axis']
            minor_list = self.props_raw['minor axis']
        self.props_proc['radius'] = [((major*minor*minor)**(1.0/3)/8) / \
                                    self.metadata['pix_per_um'] for major, minor in\
                                    zip(major_list, minor_list)]
        self.props_proc['average radius'] = np.mean(self.props_proc['radius'])

        # converts speed from pix/s to m/s (TrackedObject only computes speed in pix/s)
        pix_per_um = self.metadata['pix_per_um']
        v_m_s = [v/pix_per_um*conv.um_2_m for v in self.props_proc['speed']]
        self.props_proc['speed [m/s]'] = v_m_s
        if len(v_m_s) > 0:
            self.props_proc['average speed [m/s]'] = np.mean(v_m_s)


    ### HELPER FUNCTIONS ###
    def offscreen_centroid(self, centroid):
        """
        Estimates previous centroid assuming just offscreen opposite flow
        direction.

        Parameters
        ----------
        centroid : 2-tuple of floats
            Centroid of the object in the detected frame (row, column)
        
        Returns
        -------
        centroid_offscreen : 2-tuple of floats
            Centroid predicted of object off screen, computed by extrapolating
            in the reverse flow direction to the edge of the frame (row, column)
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
        centroid_offscreen = (row_off, col_off)

        return centroid_offscreen


    def steps_to_boundary(self, x, x_max, s, x_min=0):
        """
        Computes number of steps s to boundary (x_min or x_max).
        
        Parameters
        ----------
        x : float
            x-coordinate of object (or column)
        x_max : float
            Maximum x-value of frame (or maximum column)
        s : float
            component of reverse flow direction along horizontal access
        x_min : float, optional
            Minimum x-value of frame (or minimum column, default=0)
        """
        assert x_max >= x and x_min <= x, 'x must be in range (x_min, x_max).'
        # pointing towards minimum boundary
        if s < 0:
            n = (x_min - x) / s
        elif s > 0:
            n = (x_max - x) / s
        # if s == 0, then infinite steps are required to reach boundary
        else:
            n = float('inf')

        return n