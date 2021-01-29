"""
compare_props_finders.py compares different OpenCV functions for computing
region props of objects in a binarized image. Region props are easily calculated
in `MATLAB` or with `scikit-image`, but with OpenCV they must be computed in a
more ad hoc fashion. The two approaches are using `findContours` and
`connectedComponentsWithStats`. This script measures the computation time for
computing different region props with functions based on each of these methods.

Author: Andy Ylitalo
Date: January 29, 2021
"""

import lib



# PARAMETERS
min_size = 20 # [pixels]
input_filepath = '../input/input.txt'


###################### TEST 1: REMOVE SMALL OBJECTS ############################
# find contours method
lib.test_remove_small_objects(True, input_filepath, min_size)
# connectedComponentsWithStats method
lib.test_remove_small_objects(False, input_filepath, min_size)
