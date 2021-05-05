"""test_readin.py

Tests the functionality of the functions in the definition file `readin.py` found in the `genl` library directory.

Author: Andy Ylitalo
Date: March 23, 2021
"""

import sys
sys.path.append('../src/')

from genl import readin


# selects input file to load
input_file = '../input/input.txt'
input_params = readin.load_params(input_file)

print(input_params)
