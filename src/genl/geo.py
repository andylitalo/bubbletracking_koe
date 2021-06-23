# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:07:34 2020

@author: Andy
geo.py contains functions related to creating,
fitting, and analyzing geometric objects.
"""


import numpy as np
import scipy.optimize
import scipy.interpolate


def calc_comps(vec, axis):
    """
    Computes distances along and off axis of given vector.

    Parameters
    ----------
    vec : 2-tuple of floats
        Vector giving coordinates of point.
    axis : 2-tuple of floats
        Vector indicating axis of reference

    Returns
    -------
    comp : float
        component along the given axis
    d_off_axis : float
        Distance of vector off given axis.

    """
    # computes component of projection along axis
    comp = np.dot(vec, axis)
    # computes projection along flow axis
    proj =  comp*axis
    d_off_axis = np.linalg.norm(vec - proj)
    
    return comp, d_off_axis
    
    
def fit_circle(rows, cols, center_estimate=(0,0)):
    """
    Fit the rows and columns of points to a circle. Returns the circle's radius, center,
    and residue (a measure of error)
    """
    def calc_R(center):
        """
        Calculate the distance of each 2D point from the center (row_c, col_c) 
        """
        row_c = center[0]
        col_c = center[1]
        return np.sqrt((rows - row_c)**2 + (cols - col_c)**2)
    
    def f_2(center):
        """
        Calculate the algebraic distance between the data points and the mean
        circle centered at (row_c, col_c)
        """
        Ri = calc_R(center)
        return Ri - Ri.mean()

    center, ier = scipy.optimize.leastsq(f_2, center_estimate)
    
    Ri = calc_R(center)
    R = np.mean(Ri)
    residue   = sum((Ri - R)**2)
    return R, center, residue


def fit_ellipse(rows, cols):
    """
    Fits the rows and columns of points to an ellipse. 
    Returns the radii, center (row, col), and and angle of rotation. 
    
    Taken directly from:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    
    def fit(x, y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        return a
        
    def ellipse_center(a):
        b, c, d, f, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b - a*c
        x0 = (c*d - b*f) / num
        y0 = (a*f - b*d) / num
        row = y0 
        col = x0

        return np.array([row, col])
    
    def ellipse_angle_of_rotation(a):
        b, c, a = a[1]/2, a[2], a[0]
        return 0.5*np.arctan(2*b / (a - c))
    
    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1 = (b*b - a*c)*( (c-a)*np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
        down2 = (b*b - a*c)*( (a-c)*np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
        res1 = np.sqrt(up/down1)
        res2 = np.sqrt(up/down2)
        return np.array([res1, res2])
    
    a = fit(cols, rows)
    center = ellipse_center(a)
    theta = ellipse_angle_of_rotation(a)
    R1, R2 = ellipse_axis_length(a)

    return R1, R2, center, theta
    

def generate_ellipse(R1, R2, center, theta, N=100):
    """
    Generate an array of x and y values that lie on an ellipse with the 
    specified center, radii, and angle of rotation (theta)
    """
    t = np.linspace(0.0, 2.0*np.pi, N)
    cols = R1*np.cos(t)*np.cos(theta) - R2*np.sin(t)*np.sin(theta) + center[0]
    rows = R1*np.cos(t)*np.sin(theta) + R2*np.sin(t)*np.cos(theta) + center[1]

    return rows, cols


def generate_polygon(rows, cols, N):
    """
    Generates an array of rows and columns of points that lie evenly spaced 
    along a polygon defined by the vertices whose rows and columns are provided.
    
    The polygon is closed by returning to the first point.

    Parameters
    ----------
    rows, cols : list or numpy array
        values of rows and columns of the vertices of the polygon
    N : int
        number of points to use to create boundary
        *if N equals or exceeds the number of pixels required to make a continuous
        boundary, a continuous boundary will be returned

    Returns
    -------
    rows_polygon, cols_polygon : numpy array
        Rows and columns of the boundary of the polygon
    """
    # Add the first point to the end of the list and convert to array if needed
    if type(rows) == list or type(cols) == list:
        rows = np.array(rows + [rows[0]])
        cols = np.array(cols + [cols[0]])
    else:
        rows = np.append(rows, rows[0])
        cols = np.append(cols, cols[0])
        
    # Parameterize the arrays and interpolate
    d = [np.linalg.norm(np.array([rows[i], cols[i]]) - \
                        np.array([rows[i+1], cols[i+1]])) \
                        for i in range(len(rows)-1)]
    d = np.cumsum([0] + d)
    t = np.linspace(0, d[-1], N)
    f_rows = scipy.interpolate.interp1d(d, rows)
    f_cols = scipy.interpolate.interp1d(d, cols)
    rows_polygon = f_rows(t)
    cols_polygon = f_cols(t)
    
    return rows_polygon, cols_polygon


def generate_rectangle(rows_corners, cols_corners):
    """
    Generates an array of rows and columns of points that
    that outline a rectangle defined by the two opposing vertices provided.
    
    Parameters
    ----------
    rows_corners, cols_corners : 2-tuples
        Rows and columns of two extremal corners of the rectangle

    Returns
    -------
    rows_rect, cols_rect : numpy arrays
        Rows and columns of the points that define a continuous
        boundary around the rectangle
    """
    # extract mins and maxes
    row_min = int(min(rows_corners))
    row_max = int(max(rows_corners))
    col_min = int(min(cols_corners))
    col_max = int(max(cols_corners))
    # initialize list of values
    rows_rect = []
    cols_rect = []
    # top border
    for col in range(col_min, col_max):
        rows_rect += [row_min]
        cols_rect += [col]
    # right border
    for row in range(row_min, row_max):
        rows_rect += [row]
        cols_rect += [col_max]
    # bottom border
    for col in range(col_max, col_min, -1):
        rows_rect += [row_max]
        cols_rect += [col]
    # left border
    for row in range(row_max, row_min, -1):
        rows_rect += [row]
        cols_rect += [cols_min]

    return np.array(rows_rect), np.array(cols_rect)