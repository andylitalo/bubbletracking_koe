# bubbletracking_koe

Python code for tracking objects in a video using the [CvVidProc](https://github.com/UkoeHB/CvVidProc) 
library, which speeds up image segmentation by performing computations with OpenCV in C++
and parallelizing computations (thanks to [UkoeHB](https://github.com/UkoeHB/)!). 

While the code can be used to track any objects, it was created for high-speed microscopy of 
objects in a microfluidic stream traveling upwards of 1 m/s. In such applications where the
object speed is much faster than the frame rate, prior knowledge of the movement of the object,
such as the flow direction of the microfluidic stream, must be incorporated into the tracking
function for accurate tracking.

## Outline of Methodology

The object-tracking is achieved by
1. Compute the background by taking the median of a specified number of frames (computation parallelized in `cvvidproc` library)
2. Subtract the background from each frame
3. Apply a hysteresis threshold to identify potential objects
4. "Smooth out" objects with morphological transformations (opening, removing small objects, and filling holes)
5. Track objects using an implementation of the Hungarian algorithm with a custom distance function
6. Filter/classify objects based on morphological characteristics

Data are saved as a pickle file.

## Dependencies

 - `cvvidproc` and its dependencies (described [here](https://github.com/UkoeHB/CvVidProc))
 - `opencv` accessible to your Python installation (installation with conda forge [here](https://anaconda.org/conda-forge/opencv); see `cvvidproc` page above for further options)
 - `tkinter` (`pip install tk`)
 - `argparse` (`pip install argparse`)
 - `matplotlib` (`pip install matplotlib`)
 - `pandas` (`pip install pandas` or see extensive guide [here](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)) 

## Running Bubble Tracking

Upload your video to your desired directory in the `input` folder, specify the parameters in
`input/input.txt`, and run
```
python main.py
```
from the `src` folder.

## Known Bugs

To specify the mask (and the flow direction, if used for object-tracking), the user will use the `ui.py`
library, which includes some interactive clicking with the `tkinter` library and dialog boxes from
`ctypes`. These interactive windows require access to the user's display and likely will not work
in Jupyter notebooks or on servers without proper ssh tunneling.

## Post-processing Analysis and Validation

Often, the best validation of image processing is visual. The `analysis/highlight.py`
method will select all frames with objects and save them with the objects highlighted
(based on the image segmentation parameters given by the user) and labeled (according
to the tracking method).

Enter `analysis` and run
```
python highlight.py 
```
and the images will be saved to the `figs` subdirectory inside the folder created for
your video in the `output` folder.

## TODOs

 - Apply homographic transformation so inner stream is horizontal and mask cuts off all of the outer stream
(currently, the mask is applied as a bounding box)
 - Actually show result of masking when asking user if mask is okay
 - Break down data into objects (is there a way to automatically generate these objects during experiment
    and just fill in the Bubble data with analysis later?):
    Experiment
    --> Trial
        --> Video
            --> Bubbles
        --> Pumps
    --> MicrofluidicDevice
        --> ObservationCapillary
        --> Tubing
        --> InnerStreamCapillary