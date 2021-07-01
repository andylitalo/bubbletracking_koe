# Tests

This folder contains tests for the image-processing methods in this package.
Below, the method for running each test is described.

## Test Tracking

To test the success of object-tracking, perform the following:

```
python basic_tracking.py 1 2 3 4 5 6 -r 1 -c 1
```
This performs tests 1 through 6, replacing existing data (`-r 1`)
and checking for the presence of a mask (`-c 1`).

These six tests test the following functionalities:
1. Track single steady-speed object (single white circle travels at steady speed left-to-right)
2. Same as 1 but some frames have image on boundary to test boundary detection
3. Same as 2 but circle moves at widely varying speed left-to-right and disappears in two frames
4. Same as 2 but now there are two circles, one that enters after the other
5. Same as 2 but the circles each disappear for a frame
6. Circle travels rapidly left-to-right, then another circle emerges on the left of the frame as
the first circle disappears (should register new object). This repeats two more times. Then a
circle appears alongside the last circle. Five objects in total.

To see the results, run

```
python highlight_test_vids.py 1 2 3 4 5 6
```
and go to
```
cd ../input/test/<#>/test/figs/
```
to view results, where `<#>` is the number of the test whose results you want to view.
If you edit the input file `input/test/input_test.txt`, the destination of the saved
images may differ.