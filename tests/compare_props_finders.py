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



# tests to run
tests_to_run = [1]
# PARAMETERS
input_filepath = 'input.txt'


###################### TEST 0: REMOVE SMALL OBJECTS ############################

if 0 in tests_to_run:
    # find contours method
    ctr, time_find = lib.test_remove_small_objects(True, input_filepath)
    print('remove_small_objects took {0:.3f} ms with findContours on average over {1:d} iterations.'.format(s_2_ms*time_find, ctr))
    # connectedComponentsWithStats method
    ctr_, time_connected = lib.test_remove_small_objects(False, input_filepath)
    print('remove_small_objects took {0:.3f} ms with connectedComponentsWithStats over {1:d} iterations.'.format(s_2_ms*time_connected, ctr))

#################### TEST 1: COMPUTE REGION PROPS ##############################

if 1 in tests_to_run:
    # find contours method
    ctr, time_find = lib.test_region_props(True, input_filepath)
    print('Computing region props took {0:.3f} ms with findContours over {1:d} iterations.'.format(s_2_ms*time_find))
    # connectedComponentsWithStats method
    ctr, time_connected = lib.test_region_props(False, input_filepath)
    print('Computing region props took {0:.3f} ms with connectedComponentsWithStats over {1:d} iterations.'.format(s_2_ms*time_connected, ctr))
