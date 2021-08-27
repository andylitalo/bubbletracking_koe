"""
compare_outlier_metrics.py compares different metrics for outliers as a means
to identify appropriate thresholds for image segmentation to detect bubbles
with high fidelity.

The main idea is to assume that after subtracting the background, the remaining
pixel value distribution is composed of Gaussian noise centered around 0 and
pixels belonging to objects in the foreground. We use the parameters of the 
Gaussian distribution fitted to the pixels in object-free frames (in which
there should only be noise) to estimate thresholds for outliers.

The following methods for identifying outliers [1] are considered, wherein the 
statistics are computed from a Gaussian fit to the pixel values in a 
backgroun:
    1. > 3*sigma from the mean (sigma = standard deviation)
    2. > 5*sigma from the mean
    3. < Q1 - 1.5*IQR or > Q3 + 1.5*IQR (Q1 = 1st quartile, Q3 = 3rd quartile,
            IQR = interquartile range)
    4. modified Z-score > 3 (?)
        modified Z-score = 0.6745*(x_i - x_med) / (med(|x_i - x_med|)), 
        where x_i is the data point, x_med = med(x_i), and med() takes the 
        median [2].

Sources:
1. https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
2. Boris Iglewicz and David Hoaglin (1993), 
"Volume 16: How to Detect and Handle Outliers", The ASQC Basic References 
in Quality Control: Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

@author Andy Ylitalo
@date August 26, 2021
"""

### IMPORTS LIBRARIES


### PARAMETERS
# path to video for testing
vid_path = '../input/sd301_co2/20210207_88bar/sd301_co2_15000_001_100_0335_79_04_10.mp4' 
# path to save folder
save_dir = '../output/analysis/compare_outlier_metrics/sd301_co2_40000_001_050_0280_88_04_10/'
# histogram plots
n_bins = 100
n_bins_im = 255
hist_max = 50

### DERIVED PARAMETERS ###
n_frames = basic.count_frames(vid_path)-1
mask_path = vid_path[:-4] + '_mask.pkl'

### ARGUMENTS TO PARSE ###
def parse_args():
    """Parses arguments for script."""
    ap = argparse.ArgumentParser(
        description='Plots histograms and saves image residuals' + \
                    ' where objects are detected.')
    ap.add_argument('-f', '--framerange', nargs=2, default=(0,0),
                    metavar=('frame_start, frame_end'),
                    help='starting and ending frames to save')
    ap.add_argument('-d', '--savediff', default=0, type=int,
                    help='Saves absolute difference from bkgd instead of image if True.')
    ap.add_argument('-c', '--colormap', default='',
                     help='Specifies colormap for saving false-color images.')
    ap.add_argument('-s', '--save', default=1, type=int,
                    help='Will only save images if True.')
    ap.add_argument('-x', '--ext', default='jpg',
                    help='Extension for saving images (no ".")')
    args = vars(ap.parse_args())

    frame_start, frame_end = int(args['framerange'][0]), int(args['framerange'][1])
    save_diff = args['savediff']
    colormap = args['colormap']
    save = args['save']
    ext = args['ext']
    
    return frame_start, frame_end, save_diff, colormap, save, ext


########################## MAIN ##############################

def main():

    ### LOADS VIDEO AND PARAMS
    # reads args
    frame_start, frame_end, \
    save_diff, colormap, save, ext = parse_args()

    # loads mask data
    first_frame, _ = basic.load_frame(vid_path, 0)
    mask_data = ui.get_polygonal_mask_data(first_frame, mask_path)

    # Computes background
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
    bkgd = improc.compute_bkgd_med_thread(vid_path,
        vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
        num_frames=num_frames_for_bkgd)

    
    ### IDENTIFIES FRAMES WITHOUT OBJECTS (SYMM DISTR)
    cap = cv2.VideoCapture(vid_path)
    for f in range(frame_start, frame_end):
        frame = 
# collects pixel values and computes stats assuming Gaussian noise


### ESTIMATES THRESHOLDS BASED ON ANALYSIS METHODS


### PLOTS HISTOGRAMS OF PIXEL VALUES IN SELECTED FRAMES ALONGSIDE THRESHOLDS