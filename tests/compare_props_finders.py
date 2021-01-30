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

s_2_ms = 1000



# PARAMETERS
min_size = 20 # [pixels]
input_filepath = 'input.txt'


###################### TEST 1: REMOVE SMALL OBJECTS ############################
# find contours method
time_find = lib.test_remove_small_objects(True, input_filepath, min_size)
print('remove_small_objects took {0:.3f} ms with findContours.'.format(s_2_ms*time_find))
# connectedComponentsWithStats method
time_connected = lib.test_remove_small_objects(False, input_filepath, min_size)
print('remove_small_objects took {0:.3f} ms with connectedComponentsWithStats.'.format(s_2_ms*time_connected))
