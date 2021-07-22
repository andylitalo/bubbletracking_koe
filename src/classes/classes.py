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
        - image : (P x Q) array of uint8
            Image of object restricted to bounding box (dims P x Q)
            Likely already binarized
        - local centroid : 2-tuple of floats
            Centroid of object relative to bounding box (i.e., within image)
        - solidity : list of floats
            Area (pixels) of object divided by area of its convex hull
            Rough measure of convexity (perfectly convex has solidity = 1)
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
                          'minor axis':[], 'orientation':[], 'bbox':[], 'image':[],
                          'solidity':[], 'local centroid':[], 'on border':[]}
        # initializes storage of processed properties
        self.props_proc = {'speed':[],
                            'average area':None, 'average speed':None,
                           'average orientation':None, 'aspect ratio':[],
                           'average aspect ratio':None, 'average solidity':None,
                           'true centroids':[]
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
            print('Property {0:s} not found in object. Returning None.'.format(prop))
            return None

    def to_dict(self):
        """Sends object's stored properties to a dictionary."""
        d = {'metadata' : self.metadata,
             'props_raw' : self.props_raw,
             'props_proc' : self.props_proc}
             
        return d

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
                print('Trying to add property {0:s} not in props_raw.'.format(key))


    def process_props(self):
        """Computes processed properties, such as speed and average values."""

        # counts the number of frames
        n_frames = len(self.props_raw['frame'])

        # computes average area [pix^2]
        area = self.props_raw['area']
        self.props_proc['average area'] = self.mean_safe(area)

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
        self.props_proc['average aspect ratio'] = self.mean_safe( \
                                            self.props_proc['aspect ratio'])
        # averages solidity
        self.props_proc['average solidity'] = self.mean_safe(self.props_raw['solidity'])

        # computes average speed [pix/s]
        speed_list = self.compute_speed()
        # shifts speeds by one index (so calculation becomes backward difference)
        if len(speed_list) > 0:
            self.props_proc['speed'] = speed_list
            self.props_proc['average speed'] = np.mean(speed_list)


    #### HELPER FUNCTIONS ####

    def appears_in_consec_frames(self, n_consec=1):
        """Returns True if object appears in consecutive frames."""
        return len(np.where(np.diff(self.props_raw['frame'])==1)[0]) >= n_consec


    def compute_speed(self):
        """
        Computes speed by taking the L2-norm of the velocity computed by
        `compute_velocity()`.

        Returns
        -------
        speed_list : list of N 2-tuples of floats
            Speed at each frame where there are data. First and second are same by
            definition. Otherwise, computed with backward Euler.
        """
        v_list = self.compute_velocity()
        speed_list = [np.linalg.norm(v) for v in v_list]

        return speed_list
        

    def compute_velocity(self):
        """
        Computes velocity (speed in row and column directions) over each interval of frames.
        Assumes same velocity in the first frame as the second.

        Returns
        -------
        v_list : list of N 2-tuples of floats
            Velocity at each frame where there are data. First and second are same by
            definition. Otherwise, computed with backward Euler.
        """
        # extracts centroid coordinates (rows and columns)
        rc, cc = list(zip(*self.props_raw['centroid']))
        # computes time steps (one element shorter than rc, cc)
        dt = np.diff(self.props_raw['frame'])/self.metadata['fps']
        # computes velocity with backward Euler formula
        v_list = [ ( (rc[i+1]-rc[i])/dt[i], (cc[i+1]-cc[i])/dt[i] ) for i in range(len(dt)) ]
        # sets velocity at first frame to be the same as the second
        if len(v_list) > 0:
            v_list.insert(0, v_list[0])

        return v_list


    def mean_safe(self, vals):
        """
        Takes mean if list of values is not empty; otherwise returns None.

        Parameters
        ----------
        vals : list
            List of values to take the mean of
        
        Returns
        -------
        mean : float (or None)
            Mean of values (or None if no values in list)
        """
        if len(vals) > 0:
            return np.mean(vals)
        else:
            return None

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

        Also, "metadata" should include:
            flow_dir : 2-tuple of floats
                Normalized vector indicating direction of flow (row, col)
            pix_per_um : float
                Pixels in one micron (conversion)
            R_i : float
                Inner stream radius predicted by Stokes eqns [m]
            R_o : float
                Outer stream radius (half of inner diameter of 
                observation capillary) [m]
            v_max : float
                Predicted maximum velocity based on Stokes eqns [m/s]
            v_interf : float
                Predicted velocity at the interface of the inner stream
                based on Stokes eqns [m/s]
            d : float
                Distance along observation capillary [m]
            L : float
                Length of observation capillary [m]
            dp : float
                Pressure drop along observation capillary [Pa]
        """

        # uses TrackedObject init fn to lay foundation for the object
        super().__init__(ID, fps, frame_dim, props_raw=props_raw, **metadata)
        # adds Bubble-specific properties
        self.props_proc.update({'radius [um]' : [], 'average radius [um]' : None, 
                        'speed [m/s]' : [], 'average speed [m/s]' : None, 
                        'flow speed [m/s]' : [], 
                        'average flow speed [m/s]' : None, 
                        'inner stream' : None,
                        'solid' : None, 'circular' : None, 'oriented' : None,
                        'consecutive' : None, 'inner stream' : None,
                        'exited' : None, 'growing' : None
                        })

       
    ###### MUTATORS ########
    def classify(self,  max_aspect_ratio=10, min_solidity=0.85,
            min_size=50, max_orientation=np.pi/8, circle_aspect_ratio=1.1,
            n_consec=3):
        """
        Classifies bubble as inner or outer stream based on velocity cutoff.
        N.B.: If there is no inner stream (inner stream flow rate Q_i = 0),
        then the velocity v_interf will be computed to be nan, in which case this
        method will leave the classification as is.

        ***Currently implemented in tests/test_classification.py***

        Parameters are currently guesses. ML would be a better method for selecting them.

        Turn this into ML with the following labels (sequences of):
        - Aspect Ratio
        - Orientation
        - Area
        - Rate of area growth
        - Speed along flow direction
        - Solidity
        - *averages thereof
        - predicted max speed
        - width of inner stream
        - height

        ***Ask Chris about this! TODO
        ***CHALLENGE: how do I classify large, elongated bubbles separately? TODO

        Parameters
        ----------
        max_aspect_ratio : float, optional
            Maximum aspect ratio to classify an object as a bubble (otherwise
            sets inner stream --> -1 to signify an error)

        min_size : int, optional
            Minimum size to have a meaningful solidity and orientation.
            Default is 10 because a 3 x 4 grid with one missing edge pixel
            has a solidity of 0.917, so a bubble with a slight artifact will
            still have a sufficient solidity (> 0.9) to be classified as a bubble
            at this size.

        Returns nothing.
        """
        return
        # # computes average speed [m/s] if not done so already
        # if self.proc_props['average flow speed [m/s]'] == None:
        #     self.process_props()

        # # determines which frames are worth looking at (large enough object, not on border)
        # worth_checking = np.logical_and(
        #                         np.asarray(self.props_raw['area']) > min_size,
        #                         np.logical_not(np.asarray(self.props_raw['on border']))
        #                         )

        # # first checks if object is an artifact (marked by inner_stream = -1)
        # # not convex enough?
        # if any( np.logical_and(
        #             np.asarray(self.props_raw('solidity')) < min_solidity,
        #             worth_checking )
        #         ):
        #     self.props_proc['solid'] = False 
        # else:
        #     self.props_proc['solid']= True 

        # # too oblong?
        # if any( np.logical_and(
        #             np.asarray(self.props_proc['aspect ratio']) > max_aspect_ratio,
        #             np.asarray([row_max - row_min for row_min, _, row_max, _ in self.props_raw['bbox']]) \
        #             < self.metadata['R_i']*conv.m_2_um*self.metadata['pix_per_um'] )
        #     ):
        #     self.props_proc['circular'] = False 
        # else:
        #     self.props_proc['circular'] = True 

        # # oriented off axis (orientation of skimage.regionprops is CCW from up direction)
        # if any( np.logical_and(
        #             np.logical_and(
        #                 np.abs(np.asarray(self.props_proc['orientation']) + np.pi/2) > max_orientation,
        #                 np.abs(np.asarray(self.props_proc['orientation']) - np.pi/2) > max_orientation),
        #             worth_checking )
        #         ):
        #     self.props_proc['oriented'] = False
        # else:
        #     self.props_proc['oriented'] = True 

        # # checks that the bubble appears in a minimum number of consecutive frames
        # if not self.appears_in_consec_frames(n_consec=n_consec):
        #     self.props_proc['consecutive'] = False 
        # else:
        #     self.props_proc['consecutive'] = True

        # # flowing backwards
        # if any(np.asarray(self.props_proc['flow speed [m/s]']) < 0):
        #     self.props_proc['inner stream'] = None
        # elif any( np.asarray(self.props_proc['flow speed [m/s]']) > self.metadata['v_interf']):
        #     self.props_proc['inner stream'] = True
        # else:
        #     self.props_proc['inner stream'] = False

        # # exits properly from downstream side of frame
        # # farthest right column TODO -- determine coordinate of bbox from flow dir
        # col_max = self.props_raw['bbox'][-1][-1] 
        # # average flow speed [m/s]
        # average_flow_speed_m_s = self.props_proc['average flow speed [m/s]']
        # # number of pixels traveled per frame along flow direction
        # if average_flow_speed_m_s is not None:
        #     pix_per_frame_along_flow = average_flow_speed_m_s * conv.m_2_um * \
        #                     self.metadata['pix_per_um'] / self.metadata['fps']
        #     # checks if the next frame of the object would reach the border or if the 
        #     # last view of object is already on the border
        #     if (col_max + pix_perf_frame_along_flow < self.metadata['frame_dim'][1]) and \
        #             not self.props_raw['on border'][-1]:
        #         self.props_proc['exited'] = False 
        #     else:
        #         self.props_proc['exited'] = True
        # else:
        #     self.props_proc['exited'] = False

        # # growing over time TODO -- is `worth_checking` to strict?
        # frames = np.asarray(self.props_raw['frame'])
        # areas = np.asarray(self.props_raw['area'])
        # if len(frames[worth_checking]) > 1:
        #     growing = np.polyfit(frames[worth_checking], areas[worth_checking], 1)[0] >= 0
        # else:
        #     growing = True
        # if not growing:
        #     self.props_proc['growing'] = False
        # else:
        #     self.props_proc['growing'] = True

        # return


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

        # # processes properties for use in predicting centroid
        if len(frames) > 1:
            self.process_props()
        
        average_flow_speed_m_s = self.props_proc['average flow speed [m/s]']

        # same behavior if multiple or no centroids have been recorded
        if self.props_proc['average flow speed [m/s]'] is not None:
            centroid_prev = centroids[-1]
            f_prev = frames[-1]
            d_flow_per_frame = average_flow_speed_m_s * \
                                conv.m_2_um*self.metadata['pix_per_um'] / \
                                self.metadata['fps']
            centroid_pred = tuple(np.asarray(centroid_prev) + \
                            d_flow_per_frame * (f - f_prev) * \
                            np.asarray(self.metadata['flow_dir']))
        elif len(centroids) != 1:
            centroid_pred = super().predict_centroid(f)
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


    def process_props(self):
        """Processes data to compute processed properties, mostly averages."""
        # computes TrackedObject processed properties in the same way
        super().process_props()

        # computes radius [um]
        self.props_proc['radius [um]'] = self.estimate_radius()
        self.props_proc['average radius [um]'] = np.mean(self.props_proc['radius [um]'])

        # converts speed from pix/s to m/s (TrackedObject only computes speed in pix/s)
        pix_per_um = self.metadata['pix_per_um']
        speed_m_s = [speed/pix_per_um*conv.um_2_m for speed in self.props_proc['speed']]
        self.props_proc['speed [m/s]'] = speed_m_s
        if len(speed_m_s) > 0:
            self.props_proc['average speed [m/s]'] = np.mean(speed_m_s)

        # computes speed along flow direction [m/s]
        flow_dir = np.array(self.metadata['flow_dir'])
        v_pix_s = self.compute_velocity()
        # projects velocity onto flow direction and converts to [m/s]
        flow_speed_m_s = [np.dot(flow_dir, v)/pix_per_um*conv.um_2_m for v in v_pix_s]
        # only saves results if there are any to save
        if len(flow_speed_m_s) > 0:
            self.props_proc['flow speed [m/s]'] = flow_speed_m_s
            self.props_proc['average flow speed [m/s]'] = np.mean(flow_speed_m_s)


    ### HELPER FUNCTIONS ###

    def estimate_radius(self):
        """
        Estimates radius [um] as geometric mean of three diameters of the
        bubble, then divides result by 2 to convert from diameter to radius. 
        Assumes symmetry about major axis (due to general alignment of major
        axis along flow axis and general cylindrical symmetry around flow axis)
        such that diameter in the other two dimensions is the minor axis
        """
        # uses width and height if major and minor axes were not computed
        if (len(self.props_raw['major axis']) == 0) or \
            (len(self.props_raw['minor axis']) == 0):
            major_list = [col_max - col_min for _, col_min, _, col_max in self.props_raw['bbox']]
            minor_list = [row_max - row_min for row_min, _, row_max, _ in self.props_raw['bbox']]
        else:
            major_list = self.props_raw['major axis']
            minor_list = self.props_raw['minor axis']

        # takes geometric mean of principal axes of ellipsoid (assumes two are minor)
        # divides by 2 to convert from diameter -> radius
        radius = [((major*minor*minor)**(1.0/3)/2) / \
                                    self.metadata['pix_per_um'] for major, minor in\
                                    zip(major_list, minor_list)]
        
        return radius 


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