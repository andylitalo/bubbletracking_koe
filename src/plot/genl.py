# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:55:33 2020

@author: Andy
plot.py contains basic plotting functions
to be used within the functions of the library.
"""

import matplotlib.pyplot as plt
import numpy as np


def get_colors(cmap_name, n):
    """Returns list of colors using given colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(val) for val in np.linspace(0, 1, n)]


def legend(ax):
    """Adds legend outside box."""
    # puts legend outside of plot box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend_x = 1
    legend_y = 0.5
    plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(legend_x, legend_y))


def no_ticks(image):
    """
    This removes tick marks and numbers from the axes of the image and fills
    up the figure window so the image is easier to see.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.axis('image')
    plt.tight_layout(pad=0)
