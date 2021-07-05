"""
testclasses.py contains classes and modifications of classes
used for running program tests.

Author: Andy Ylitalo
Date: July 5, 2021
"""


# imports classes to build from
import sys
sys.path.append('../src/')
from classes.classes import TrackedObject, Bubble


class BubbleTest(Bubble):
    """
    Same as Bubble class from src/classes/classes.py, but with
    some additional methods used for tests.
    """
    def __init__(self, ID, fps, frame_dim, props_raw=[], **metadata):
        """
        See __init__ for Bubble
        """
        # uses Bubble init fn to lay foundation for the object
        super().__init__(ID, fps, frame_dim, props_raw=props_raw, **metadata)
        # adds some additional properties
        self.props_proc['solid'] = None
        self.props_proc['circular'] = None 
        self.props_proc['oriented'] = None 
        self.props_proc['consecutive'] = None
        
    ### HELPER FUNCTIONS ###
    
    def classify_test(self, v_interf, max_aspect_ratio=10, min_solidity=0.9,
                min_size=10, max_orientation=np.pi/10, circle_aspect_ratio=1.1,
                n_consec=3):
        """
        Classifies Bubble objects into different categories. Still testing to see
        what classifications are most important.

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
        """
        # computes average speed [m/s] if not done so already
        if self.props_proc['average flow speed [m/s]'] == None:
            self.process_props()

        # first checks if object is an artifact (marked by inner_stream = -1)
        # not convex enough?
        if any( np.logical_and(
                    np.asarray(self.props_raw['solidity']) < min_solidity,
                    np.asarray(self.props_raw['area']) > min_size )
                ):
            self.props_proc['solid'] = False 
        else:
            self.props_proc['solid']= True 

        # too oblong?
        if any( np.logical_and(
                    np.asarray(self.props_raw['aspect ratio']) > max_aspect_ratio,
                    np.asarray([row_max - row_min for row_min, _, row_max, _ in self.props_raw['bbox']]) \
                    < self.metadata['R_i']*conv.m_2_um*self.metadata['pix_per_um']
            )):
            self.props_proc['circular'] = False 
        else:
            self.props_proc['circular'] = True 

        # oriented off axis (orientation of skimage.regionprops is CCW from up direction)
        if any( np.logical_and(
                    np.abs(np.asarray(self.props_raw['orientation']) + np.pi/2) < max_orientation,
                    np.asarray(self.props_raw['area']) > min_size
                )):
            self.props_proc['oriented'] = False
        else:
            self.props_proc['oriented'] = True 

        # checks that the bubble appears in a minimum number of consecutive frames
        if not self.appears_in_consec_frames(n_consec=n_consec):
            self.props_proc['consecutive'] = False 
        else:
            self.props_proc['consecutive'] = True

        # flowing backwards or flowing too fast
        if any( np.logical_or(
                    np.asarray(self.props_proc['flow speed [m/s]']) < 0,
                    np.asarray(self.props_proc['speed [m/s]'] > 3*self.metadata['v_max'])
            )):
            self.props_proc['inner stream'] = -1
        elif any( np.asarray(self.props_proc['flow speed [m/s]']) > v_interf):
            self.props_proc['inner stream'] = 1
        else:
            self.props_proc['inner stream'] = 0

    def appears_in_consec_frames(self, n_consec=1):
    """Returns True if object appears in consecutive frames."""
    return len(np.where(np.diff(self.frame)==1)[0]) >= n_consec
