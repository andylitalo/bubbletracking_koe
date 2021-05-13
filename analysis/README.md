`hist_pix.py`identifies thresholds for detecting bubbles based on aggregate 
data for an image (mean, standard deviation, minimum, etc.).
Saves images and histograms of frames that pass different detection criteria
for user analysis. 
RESULTS
-Otsu's method gives roughly the same threshold for the minimum value as
k-means, both of which are too low to detect small, faint bubbles
-Instead, using a ratio of the standard deviation appears to be more useful
as a threshold for detecting bubbles
