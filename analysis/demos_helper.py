"""
demos_helper.py contains small helper functions
utilized by the more complex methods of demos.py.

*Note: requires Bokeh plotting library (tested with v1.4.0).

Author: Andy Ylitalo
Date: June 11, 2021
"""

# Python libraries
import os
import numpy as np
import cv2

# 3rd party libraries
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.models import ColumnDataSource, LabelSet, Rect, Line
# image-processing
import skimage.measure

# custom libraries
import sys
sys.path.append('../src')
import cvimproc.improc as improc
import config as cfg


def add_colorbar(p, palette, low, high, cb_fs=14):
    """
    Adds colorbar to a Bokeh figure.
    
    Parameters
    ----------
    p : Bokeh figure
        figure that will have a colorbar added to it
    palette : string
        Name of color palette (e.g., 'Turbo256' for 256 colors)
    low, high : ints
        lowest and highest values included in colorbar
    cb_fs : int, opt (default=14)
        font size of colorbar labels 

    Returns
    -------
    p : Bokeh figure
        figure, now with colorbar
    """
    # creates color mapper
    color_mapper = LinearColorMapper(palette=palette, low=low, high=high)
    # creates colorbar
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=10)
    # formats font size
    color_bar.major_label_text_font_size = '{0:d}pt'.format(cb_fs)
    # adds color bar to figure
    p.add_layout(color_bar, 'left')
    
    return p, color_mapper


def bbox_obj(frame, objs, ID, f, color, label_key=None, offset=10):
    """
    """
    # checks that the ID number is included in the dictionary of objects
    if ID not in objs.keys():
        return
    
    # shows number ID of object in image
    # first finds the centroid of the object
    centroid = objs[ID].get_prop('centroid', f)
    # converts centroid from (row, col) to (x, y) for open-cv
    x = int(centroid[1])
    y = int(centroid[0])
    
    # gets label
    if label_key is not None:
        label = label_key[ID]
    else:
        label = str(ID)
        
    # prints label near object
    cv2.putText(img=frame, text=label, org=(x, y-offset),
                            fontFace=0, fontScale=1, color=color,
                            thickness=2)
    
    # bounding box is (min_row, min_col, max_row, max_col)
    bbox = objs[ID].get_prop('bbox', f)
    min_row, min_col, max_row, max_col = bbox
    # prints bounding box
    cv2.rectangle(frame, (min_col, min_row), (max_col, max_row),
                          color, 2)
    # prints dot at centroid
    cv2.circle(frame, (x, y), 4, color, -1)
            
    return


def get_dirs(p):
    """
    Gets directories used by demos.process_video()
    
    Parameters
    ----------
    p : dictionary
        Parameters of image processing
        
    Returns
    -------
    vid_path, vid_dir, data_dir, figs_dir : strings
        vid_path = filepath to video to process; vid_dir = directory
        containing video to process; data_dir = directory to which to 
        save data (.pkl file); figs_dir = directory to which to save figures.
    """
    # defines filepath to video
    vid_path = os.path.join(p['input_dir'], p['vid_subdir'], p['vid_name'])
    # defines directory to video data and figures
    vid_dir = os.path.join(p['vid_subdir'], p['vid_name'][:-4], p['input_name'])
    data_dir = os.path.join(p['output_dir'], vid_dir, cfg.data_subdir)
    figs_dir = os.path.join(p['output_dir'], vid_dir, cfg.figs_subdir)
    
    return vid_path, vid_dir, data_dir, figs_dir


def label_and_measure_objs(p, frame_labeled, objects, pix_per_length):
    """
    """
    # initializes lists for storing coordinates of bounding boxes and axes    
    x = []
    y = []
    w = []
    h = []
    name = []
    x_mins = []
    y_mins = []
    x_majs = []
    y_majs = []
    
    for i in objects:
        obj = objects[i]
        
        # overlays bounding box
        row_min, col_min, row_max, col_max = obj['bbox']
        x += [(col_max + col_min) / 2 / pix_per_length]
        y += [(row_max + row_min) / 2 / pix_per_length]
        w += [(col_max - col_min) / pix_per_length]
        h += [(row_max - row_min) / pix_per_length]
        name += [str(i)]
        
        yc, xc = obj['centroid']
        orientation = obj['orientation']
        semiminor_axis = obj['semiminor axis']
        semimajor_axis = obj['semimajor axis']
        
        # converts from pixels to microns
        xc /= pix_per_length
        yc /= pix_per_length
        semiminor_axis /= pix_per_length
        semimajor_axis /= pix_per_length
        
        x_mins += [(xc, xc + np.cos(orientation) * semiminor_axis)]
        y_mins += [(yc, yc - np.sin(orientation) * semiminor_axis)]
        x_majs += [(xc, xc - np.sin(orientation) * semimajor_axis)]
        y_majs += [(yc, yc - np.cos(orientation) * semimajor_axis)]
    
    # collects data
    source = ColumnDataSource(dict(x=x, y=y, w=w, h=h, name=name))
    
    # plots bounding box around each object
    glyph = Rect(x="x", y="y", width="w", height="h", fill_alpha=0, line_color='white')
    p.add_glyph(source, glyph)
    
    # plots major and minor axis of each object
    for x_min, y_min, x_maj, y_maj in zip(x_mins, y_mins, x_majs, y_majs):
        line_source = ColumnDataSource(dict(x_min=x_min, y_min=y_min, x_maj=x_maj, y_maj=y_maj))
        minor = Line(x="x_min", y="y_min", line_width=1, line_alpha=0.6)
        major = Line(x="x_maj", y="y_maj", line_width=1, line_alpha=0.6)
        p.add_glyph(line_source, minor)
        p.add_glyph(line_source, major)
    
    # labels objects
    labels = LabelSet(x='x', y='y', text='name', x_offset=-6, y_offset=-8, source=source, render_mode='canvas')
    p.add_layout(labels)

    return p
    

def set_bokeh_fig_fonts(p, t_fs, ax_fs, tk_fs):
    """
    """
    p.title.text_font_size = '{0:d}pt'.format(t_fs)
    p.xaxis.axis_label_text_font_size = '{0:d}pt'.format(ax_fs)
    p.xaxis.major_label_text_font_size = '{0:d}pt'.format(tk_fs)
    p.yaxis.axis_label_text_font_size = '{0:d}pt'.format(ax_fs)
    p.yaxis.major_label_text_font_size = '{0:d}pt'.format(tk_fs)
    
    return p


def store_region_props(frame_labeled, num, width_border=2):
    """
    """
    # identifies the different objects in the frame
    region_props = skimage.measure.regionprops(frame_labeled)
    objects = {}
    for i, props in enumerate(region_props):
        # creates dictionary of object properties for one frame
        # for all the measured properties, see:
        # https://scikit-image.org/docs/dev/api/skimage.measure.html
        obj = {}
        obj['centroid'] = props.centroid
        obj['area'] = props.area
        obj['orientation'] = props.orientation
        # really the semi-major and semi-minor axes
        obj['semimajor axis'] = props.major_axis_length
        obj['semiminor axis'] = props.minor_axis_length
        obj['bbox'] = props.bbox # (row_min, col_min, row_max, col_max)
        obj['frame'] = num
        obj['on border'] = improc.is_on_border(props.bbox,
              frame_labeled, width_border)
        obj['solidity'] = props.solidity
        obj['aspect ratio'] = obj['semimajor axis'] / obj['semiminor axis']
        # adds dictionary for this object to list of objects in current frame
        objects[i] = obj
        
    return objects