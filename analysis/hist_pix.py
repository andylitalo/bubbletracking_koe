"""
hist_pix.py is used to plot histograms of pixel values of the
signed difference from the background (median). It first computes
the mean and standard deviation of the signed difference from the
background, then uses that information to classify images based on 
mean, standard deviation, and minimum value of the signed difference.

Goal: for the classifications above to correspond to bubble detection.

@author Andy Ylitalo
@date May 5, 2021
"""
### IMPORT LIBRARIES ###
# add source folder to path for imports
import sys
sys.path.append('../src/')

# import Python libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# imports 3rd-party libraries
import cv2
import PIL.Image
import skimage.filters
from sklearn.cluster import KMeans

# imports custom libraries
import cvimproc.improc as improc
import cvimproc.basic as basic
import cvimproc.ui as ui
import cvimproc.mask as mask
import statlib as stat


### PARAMETERS ###
# path to video for testing
vid_path = '../input/sd301_co2/20210403_76bar/sd301_co2_40000_001_050_0280_88_04_10.mp4'
# path to save folder
save_path = '../output/hist_pix/sd301_co2_40000_001_050_0280_88_04_10/'
# extension for images to save
ext = 'jpg'
# analysis parameters
num_frames_for_bkgd = basic.count_frames(vid_path)-1
end = basic.count_frames(vid_path)-1
# histogram plots
n_bins = 100
n_bins_im = 255
hist_max = 50
# thresholds
th_mean = 0.25
th_stdev = 3.2
# bubbles must have a min val < this many stdevs from mean min val
n_sigma = 5

### DERIVED PARAMETERS ###
mask_path = vid_path[:-4] + '_mask.pkl'

### ARGUMENTS TO PARSE ###
def parse_args():
    """Parses arguments for script."""
    ap = argparse.ArgumentParser(
        description='Plots histograms and saves image residuals' + \
                    ' where bubbles are detected.')
    ap.add_argument('-f', '--framerange', nargs=2, default=(0,0),
                    metavar=('frame_start, frame_end'),
                    help='starting and ending frames to save')
    ap.add_argument('-s', '--stopearly', default=False, type=bool,
                    help='Stops analysis before saving images if True.')
    ap.add_argument('-d', '--savediff', default=False, type=bool,
                    help='Saves absolute difference from bkgd instead of image if True.')
    ap.add_argument('-c', '--colormap', default='', type=string,
                    help='Specifies colormap for saving false-color images.')
    args = vars(ap.parse_args())

    frame_start, frame_end = int(args['framerange'][0]), int(args['framerange'][1])
    stop_early = args['stopearly']
    save_diff = args['savediff']
    colormap = args['colormap']
    
    return frame_start, frame_end, stop_early, save_diff, colormap

def proc_stats(vid_path, bkgd, end):
    """Processes statistics of the images in the video."""
    # initializes lists to store data
    mean_list = []
    stdev_list = []
    mean_sq_list = []
    min_val_list = []

    # load frames and process
    cap = cv2.VideoCapture(vid_path)
    f = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        signed_diff = frame.astype(int) - bkgd.astype(int)
        mean_list += [np.mean(signed_diff)]
        mean_sq_list += [np.mean(signed_diff**2)]
        stdev_list += [np.std(signed_diff)]
        min_val_list += [np.min(signed_diff)]

        if not ret or f >= end:
            break
        
        f += 1

    cap.release()

    return mean_list, mean_sq_list, stdev_list, min_val_list




### COMPUTATIONS ###

def main():
    
    frame_start, frame_end, stop_early, save_diff, colormap = parse_args()

    # creates color map
    if len(colormap) > 0:
        cm = plt.cm.get_cmap(colormap, 2*255+1)
        
    # loads mask data
    first_frame, _ = basic.load_frame(vid_path, 0)
    mask_data = ui.get_polygonal_mask_data(first_frame, mask_path)

    # Compute background
    row_lo, _, row_hi, _ = mask.get_bbox(mask_data)
    bkgd = improc.compute_bkgd_med_thread(vid_path,
        vid_is_grayscale=True,  #assume video is already grayscale (all RGB channels are the same)
        num_frames=num_frames_for_bkgd)

    # Prints frames and histograms of desired frames
    desired_frames = np.arange(frame_start, frame_end)
    for f in desired_frames:
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, bkgd)
        if save_diff:
            im = PIL.Image.fromarray(diff)
        else:
            im = PIL.Image.fromarray(frame)
        im.save(os.path.join(save_path, '{0:d}.{1:s}'.format(f, ext)))
        # plots histogram
        signed_diff = frame.astype(int) - bkgd.astype(int)
        plt.figure()
        _ = plt.hist(signed_diff.flatten(), n_bins_im, histtype='step')
        plt.ylim(0, hist_max)
        plt.title('{0:d}'.format(f))
        plt.savefig(os.path.join(save_path, 'hist_{0:d}.{1:s}'.format(f, ext)))
        plt.close()
        if len(colormap) > 0:
            # scales brightness of pixels to saturation (or near it)
            scale = int(255 / np.max(np.abs(signed_diff)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fc = ax.pcolormesh(scale*signed_diff, cmap=cmap, rasterized=True, vmin=-255, vmax=255)
            fig.colorbar(fc, ax=ax)
            plt.savefig(os.path.join(save_path, 'fc_{0:d}.{1:s}'.format(f, ext)))
            plt.close()

    # computes statistics
    mean_list, mean_sq_list, stdev_list, min_val_list = stat.proc_stats(vid_path, 
                                                                bkgd, end)

    # computes stats
    mu = np.mean(mean_list)
    sigma = np.sqrt(np.mean(mean_sq_list) - mu**2)
    sigma_approx = np.mean(stdev_list)
    print('Mean mu = {0:f}, stdev sigma = {1:f}.'.format(mu, sigma))
    print('Approximate sigma is {0:f}.'.format(sigma_approx))
    # stats of the minimum values
    min_val_arr = np.asarray(min_val_list)
    mu_min = np.mean(min_val_arr)
    sigma_min = np.std(min_val_arr)
    print('Mean min = {0:.2f}; std min = {1:.2f}; mu - {2:.1f}*sigma = {3:.2f}'.format(mu_min,
                        sigma_min, n_sigma, mu_min - n_sigma*sigma_min))

    # PLOTS HISTOGRAMS
    # MEAN
    plt.figure()
    _ = plt.hist(mean_list, n_bins, histtype='step')
    plt.ylim([0, hist_max])
    plt.plot([mu, mu], [0, hist_max], 'k-', label='overall mean = {0:.2f}'.format(mu))
    plt.title('Mean')
    plt.legend()
    # saves figure
    plt.savefig(os.path.join(save_path, 'mean_hist.png'))

    # STANDARD DEVIATION
    plt.figure()
    _ = plt.hist(stdev_list, n_bins, histtype='step')
    plt.ylim([0, hist_max])
    plt.plot([sigma, sigma], [0, hist_max], 'k-', label='overall stdev = {0:.2f}'.format(sigma))
    plt.title('Standard Deviation')
    plt.legend()
    # saves figure
    plt.savefig(os.path.join(save_path, 'stdev_hist.png'))

    # MINIMUM
    plt.figure()
    _ = plt.hist(min_val_list, n_bins, histtype='step')
    plt.title('Minimum')
    plt.ylim([0, hist_max])
    plt.plot([mu_min, mu_min], [0, hist_max], 'k-', label='mu')
    threesig = mu_min - 3*sigma_min
    fivesig = mu_min - 5*sigma_min
    plt.plot([threesig, threesig], [0, hist_max], 'b--', 
                    label=r'$\mu - 3\sigma$' + '{0:.2f}'.format(threesig))
    plt.plot([fivesig, fivesig], [0, hist_max], 'g--', 
                    label=r'$\mu - 5\sigma$' + '{0:.2f}'.format(fivesig))
    plt.legend()
    # saves figure
    plt.savefig(os.path.join(save_path, 'min_hist.png'))

    if stop_early:
        return 0

    ####################### ROUND 2 OF ANALYSIS #############################
    # now that we know the mean and standard deviation of each image, we can
    # save images based on how their means and standard deviations compare to
    # the overall values
    print('Mean and standard deviation computed. Now save individual images and histograms.')


    #th_min = -n_sigma*sigma
    th_min = mu_min - n_sigma*sigma_min # removes mean since we want to compare to median (= 0)
    # uses Otsu's method to identify threshold b/w min val for bubble and for no bubble
    th_min_otsu = skimage.filters.threshold_otsu(min_val_arr)
    # clusters minima into 2 -- use boundary as threshold?
    inds_clusters = KMeans(n_clusters=2, random_state=1).fit_predict(min_val_arr.reshape(-1, 1))
    i_low_cluster = inds_clusters[np.argmin(min_val_arr)]
    th_min_kmeans_low = np.max(min_val_arr[inds_clusters==i_low_cluster])
    i_high_cluster = inds_clusters[np.argmax(min_val_arr)]
    th_min_kmeans_high = np.min(min_val_arr[inds_clusters==i_high_cluster])



    # reports results
    print('Threshold for minimum is {0:.1f} for mu_min - {1:.1f}*sigma_min.' \
                            .format(th_min, n_sigma))
    print('Otsu threshold for minimum is {0:.1f}'.format(th_min_otsu))
    print('K-means low threshold for minimum is {0:.1f}'.format(th_min_kmeans_low))
    print('K-means high threshold for minimum is {0:.1f}'.format(th_min_kmeans_high))


    # clusters w/ kernal density estimation
    from sklearn.neighbors import KernelDensity
    from scipy.signal import argrelextrema
    # reshapes data for Kernel Density Estimation
    X = min_val_arr.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X)
    # range of samples
    samples = np.arange(-100, -10)
    # expected density of samples
    expected = kde.score_samples(samples.reshape(-1, 1))
    # indices of minima and maxima of expected density
    mi, ma = argrelextrema(expected, np.less)[0], argrelextrema(expected, np.greater)[0]
    # minimum threshold is the highest minimum of the kernel density
    th_min_kde = np.max(samples[mi])
    print('Threshold from KDE is {0:.1f}.'.format(th_min_kde))


    # load frames and process
    cap = cv2.VideoCapture(vid_path)
    f = 0

    while(cap.isOpened()):
        # loads frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, bkgd)
        
        signed_diff = frame.astype(int) - bkgd.astype(int)
        mean = np.mean(signed_diff)
        stdev = np.std(signed_diff)
        min_val = np.min(signed_diff)

        # saves image if a threshold is exceeded
        sub_dirs = []
    #    if mean > th_mean:
    #        sub_dirs += ['above_mean']
    #    if mean < -th_mean:
    #        sub_dirs += ['below_mean']
        if stdev > th_stdev:
            sub_dirs += ['above_stdev']
        if min_val < th_min:
            sub_dirs += ['below_min']

        # if both below min and above stdev, save in intersxn dir
        # this combo seems like a good test of bubbles
    #    if 'below_min' in sub_dirs and 'above_stdev' in sub_dirs:
    #        sub_dirs += ['stdev_min']

        for sub_dir in sub_dirs:
            print('saving frame {0:d}'.format(f))

            if save_diff:
                im = PIL.Image.fromarray(diff)
            else:
                im = PIL.Image.fromarray(frame)
            im.save(os.path.join(save_path, sub_dir, '{0:d}.{1:s}'.format(f, ext)))
            # plotting as step is faster
            plt.figure()
            _ = plt.hist(signed_diff.flatten(), n_bins_im, histtype='step')
            plt.ylim(0, hist_max)
            plt.title('f{0:d}; mean = {1:.2f}; std = {2:.2f}; min = {3:d}'.format(f, \
                                        mean, stdev, int(min_val)))
            plt.savefig(os.path.join(save_path, sub_dir, \
                                        'hist_{0:d}.{1:s}'.format(f, ext)))
            plt.close()

        if not ret or f >= end:
            break
        
        f += 1

    cap.release()

    return 0


if __name__ == '__main__':
    main()

"""  
    # highlight bubbles and extract images at each step
    proc_ims = improc.highlight_bubble_hyst_thresh(
                            frame, bkgd, th, th_lo, th_hi,
                            min_size_hyst, min_size_th, width_border, 
                            selem, mask_data, ret_all_steps=True)
    
    im_diff, thresh_bw_1, bubble_1, thresh_bw_2, bubble_2, bubble = proc_ims
"""
