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
    
    
def fit_circle(x,y,center_estimate=(0,0)):
    """
    Fit the x and y points to a circle. Returns the circle's radius, center,
    and residue (a measure of error)
    """
    def calc_R(center):
        """
        Calculate the distance of each 2D point from the center (xc, yc) 
        """
        xc = center[0]
        yc = center[1]
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(center):
        """
        Calculate the algebraic distance between the data points and the mean
        circle centered at (xc, yc)
        """
        Ri = calc_R(center)
        return Ri - Ri.mean()

    center, ier = scipy.optimize.leastsq(f_2,center_estimate)
    
    Ri = calc_R(center)
    R = np.mean(Ri)
    residue   = sum((Ri - R)**2)
    return R, center, residue


def fit_ellipse(x,y):
    """
    Fit the x and y points to an ellipse. Returns the radii, center,
    and and angle of rotation. Taken directly from:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    
    def fit(x,y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        return a
        
    def ellipse_center(a):
        b,c,d,f,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return np.array([x0,y0])
    
    def ellipse_angle_of_rotation(a):
        b,c,a = a[1]/2, a[2], a[0]
        return 0.5*np.arctan(2*b/(a-c))
    
    def ellipse_axis_length(a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    a = fit(x,y)
    center = ellipse_center(a)
    theta = ellipse_angle_of_rotation(a)
    [R1,R2] = ellipse_axis_length(a)

    return R1, R2, center, theta
    

def generate_ellipse(R1,R2,center,theta,N=100):
    """
    Generate an array of x and y values that lie on an ellipse with the 
    specified center, radii, and angle of rotation (theta)
    """
    t = np.linspace(0.0,2.0*np.pi,N)
    x = R1*np.cos(t)*np.cos(theta) - R2*np.sin(t)*np.sin(theta) + center[0]
    y = R1*np.cos(t)*np.sin(theta) + R2*np.sin(t)*np.cos(theta) + center[1]
    return x,y


def generate_polygon(x,y,N):
    """
    Generate an array of x and y values that lie evenly spaced along a polygon
    defined by the x and y values where it is assumed that the first value
    is also the last value to close the polygon
    """
    # Add the first point to the end of the list and convert to array if needed
    if type(x) == list:
        x = np.array(x + [x[0]])
        y = np.array(y + [y[0]])
    else:
        x = np.append(x,x[0])
        y = np.append(y,y[0])
        
    # Parameterize the arrays and interpolate
    d = [np.linalg.norm(np.array([x[i],y[i]])-np.array([x[i+1],y[i+1]]))
         for i in range(len(x)-1)]
    d = np.cumsum([0]+d)
    t = np.linspace(0,d[-1],N)
    fx = scipy.interpolate.interp1d(d,x)
    fy = scipy.interpolate.interp1d(d,y)
    x = fx(t)
    y = fy(t)
    
    return x,y


def generate_rectangle(xVert, yVert):
    """
    Generate an array of x and y values that outline a rectangle defined by the
    two opposing vertices given in xVert and yVert
    xVert, yVert: each is a numpy array of two values, a min and a max, not 
    necessarily in that order
    """
    # extract mins and maxes
    xMin = int(min(xVert))
    xMax = int(max(xVert))
    yMin = int(min(yVert))
    yMax = int(max(yVert))
    # initialize list of values
    X = []
    Y = []
    # top border
    for x in range(xMin,xMax):
        X += [x]
        Y += [yMin]
    # right border
    for y in range(yMin, yMax):
        X += [xMax]
        Y += [y]
    # bottom border
    for x in range(xMax, xMin, -1):
        X += [x]
        Y += [yMax]
    # left border
    for y in range(yMax, yMin, -1):
        X += [xMin]
        Y += [y]
    return np.array(X), np.array(Y)


def v_sphere(R):
    """Computes the volume of a sphere given the radius R."""
    return (4*np.pi/3) * (R**3)
