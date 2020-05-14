#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:49:12 2020

@author: btreece
"""

import numpy as np

###############################################################################
# First Derivative ############################################################
###############################################################################

def __D1_Five_Point_Stencil(p, h):
    """ Apply a five point stencil to p[0:5], at p[2], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    
    return ( 8.0 * (p[3] - p[1]) - (p[4] - p[0]) ) / (12.0 * h)

def __D1_Five_Point_Stencil_Assymmetric_F(p, h):
    """ Apply a five point stencil to p[0:5], at p[1], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (-10.0*p[1] + 18.0*p[2] - 3.0*p[0] - 6.0*p[3] + p[4]) / (12.0 * h)

def __D1_Five_Point_Stencil_Assymmetric_B(p, h):
    """ Apply a five point stencil to p[0:5], at p[3], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (10.0*p[3] + 3.0*p[4] - 18.0*p[2] + 6.0*p[1] - p[0]) / (12.0 * h)

def __D1_Five_Point_Stencil_Forward(p, h):
    """ Apply a five point stencil to p[0:5], at p[0], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (-25.0*p[0] + 48.0*p[1] - 36.0*p[2] + 16.0*p[3] - 3*p[4]) / (12.0 * h)

def __D1_Five_Point_Stencil_Backward(p, h):
    """ Apply a five point stencil to p[0:5], at p[4], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (25.0*p[4] - 48.0*p[3] + 36.0*p[2] - 16.0*p[1] + 3*p[0]) / (12.0 * h)


###############################################################################
def D1_5(data, step, axis = -1):
    """Calculate the derivative of an array whose domain is equally spaced. One
    or two dimensional arrays are supported.

    Parameters
    ----------
    data : numpy ndarray
        The data on which the derivative is to be applied.
    step : float
        The spacing between points in the domain of the data; assumed to be
        equally spaced.
    axis : integer, optional
        For multi-dimensional arrays of data the axis over which a single data
        set varies.

    Returns
    -------
    out : ndarray
        Array containing the data after application of the derivative.

    Examples -- TAKEN FROM numpy.asarray
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.asarray(a, dtype=np.float32) is a
    True
    >>> np.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True

    """
    
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be of type numpy.ndarray")
    data_shape = np.shape(data)
    if not ((len(data_shape) == 1) or (len(data_shape) == 2)):
        raise ValueError("'data' must be either 1 or 2 dimensional")
    
    if axis == -1:
        axis = len(data_shape) - 1
    elif ((axis == 1) and (len(data_shape) != 2)):
        raise ValueError("'axis' can only be equal to 1 for two-dimensional data")
    elif axis==0:
        pass
    else:
        raise ValueError("'axis' can only have the value -1, 0, or 1")
    
    bRotated = False
    if ((len(data_shape) == 2) and (axis == 0)):
        data = data.T
        data_shape = np.shape(data)
        bRotated = True
    
    if data_shape[-1] < 5:
        raise Exception("Cannot perform a stencil on fewer than 5 points!")
    elif data_shape[-1] < 10:
        raise Warning("You're using at least half your data to calculate the "+
                      "derivative at each point!")
    
    out = -100.0 + 0.0*np.copy(data)
    
    if len(data_shape) == 1:
        out[0] = __D1_Five_Point_Stencil_Forward(data[0:5], step)
        out[1] = __D1_Five_Point_Stencil_Assymmetric_F(data[0:5], step)
        out[-2] = __D1_Five_Point_Stencil_Assymmetric_B(data[-5:], step)
        out[-1] = __D1_Five_Point_Stencil_Backward(data[-5:], step)
        for i in range(2, data_shape[0]-2):
            out[i] = __D1_Five_Point_Stencil(data[i-2:i+3], step)
    if len(data_shape) == 2:
        for j in range(data_shape[0]):
            out[j,0] = __D1_Five_Point_Stencil_Forward(data[j,0:5], step)
            out[j,1] = __D1_Five_Point_Stencil_Assymmetric_F(data[j,0:5], step)
            out[j,-2] = __D1_Five_Point_Stencil_Assymmetric_B(data[j,-5:], step)
            out[j,-1] = __D1_Five_Point_Stencil_Backward(data[j,-5:], step)
            for i in range(2, data_shape[1]-2):
                out[j,i] = __D1_Five_Point_Stencil(data[j,i-2:i+3], step)

    #######
    if bRotated:
        out = out.T
        
    return out

###############################################################################
# Second Derivative ###########################################################
###############################################################################

def __D2_Five_Point_Stencil(p, h):
    """ Apply a second derivative five point stencil to p[0:5], at p[2], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    
    return ( -30.0*p[2] + 16.0*(p[1] + p[3]) - (p[0] + p[4]) ) / (12.0 * (h**2.0))

def __D2_Five_Point_Stencil_Assymmetric_F(p, h):
    """ Apply a second derivative five point stencil to p[0:5], at p[1], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (11.0*p[0] - 20.0*p[1] + 6.0*p[2] + 4.0*p[3] - 1.0*p[4]) / (12.0 * (h**2.0))

def __D2_Five_Point_Stencil_Assymmetric_B(p, h):
    """ Apply a second derivative five point stencil to p[0:5], at p[3], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (-1.0*p[0] + 4.0*p[1] + 6.0*p[2] - 20.0*p[3] + 11.0*p[4]) / (12.0 * (h**2.0))

def __D2_Five_Point_Stencil_Forward(p, h):
    """ Apply a second derivative five point stencil to p[0:5], at p[0], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (35.0*p[0] - 104.0*p[1] + 114.0*p[2] - 56.0*p[3] + 11.0*p[4]) / (12.0 * (h**2.0))

def __D2_Five_Point_Stencil_Backward(p, h):
    """ Apply a second derivative five point stencil to p[0:5], at p[4], with step h.
    Parameters
    ----------
    
    p : iterable
    h : float
    
    Returns
    -------
    out : float
    """
    return (11.0*p[0] - 56.0*p[1] + 114.0*p[2] - 104.0*p[3] + 35.0*p[4]) / (12.0 * (h**2.0))


###############################################################################
def D2_5(data, step, axis = -1):
    """Calculate the second derivative of an array whose domain is equally
       spaced. One or two dimensional arrays are supported.

    Parameters
    ----------
    data : numpy ndarray
        The data on which the derivative is to be applied.
    step : float
        The spacing between points in the domain of the data; assumed to be
        equally spaced.
    axis : integer, optional
        For multi-dimensional arrays of data the axis over which a single data
        set varies.

    Returns
    -------
    out : ndarray
        Array containing the data after application of the derivative.

    Examples -- TAKEN FROM numpy.asarray
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.asarray(a, dtype=np.float32) is a
    True
    >>> np.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True

    """
    
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be of type numpy.ndarray")
    data_shape = np.shape(data)
    if not ((len(data_shape) == 1) or (len(data_shape) == 2)):
        raise ValueError("'data' must be either 1 or 2 dimensional")
    
    if axis == -1:
        axis = len(data_shape) - 1
    elif ((axis == 1) and (len(data_shape) != 2)):
        raise ValueError("'axis' can only be equal to 1 for two-dimensional data")
    elif axis==0:
        pass
    else:
        raise ValueError("'axis' can only have the value -1, 0, or 1")
    
    bRotated = False
    if ((len(data_shape) == 2) and (axis == 0)):
        data = data.T
        data_shape = np.shape(data)
        bRotated = True
    
    if data_shape[-1] < 5:
        raise Exception("Cannot perform a stencil on fewer than 5 points!")
    elif data_shape[-1] < 10:
        raise Warning("You're using at least half your data to calculate the "+
                      "derivative at each point!")
    
    out = -100.0 + 0.0*np.copy(data)
    
    if len(data_shape) == 1:
        out[0] = __D2_Five_Point_Stencil_Forward(data[0:5], step)
        out[1] = __D2_Five_Point_Stencil_Assymmetric_F(data[0:5], step)
        out[-2] = __D2_Five_Point_Stencil_Assymmetric_B(data[-5:], step)
        out[-1] = __D2_Five_Point_Stencil_Backward(data[-5:], step)
        for i in range(2, data_shape[0]-2):
            out[i] = __D2_Five_Point_Stencil(data[i-2:i+3], step)
    if len(data_shape) == 2:
        for j in range(data_shape[0]):
            out[j,0] = __D2_Five_Point_Stencil_Forward(data[j,0:5], step)
            out[j,1] = __D2_Five_Point_Stencil_Assymmetric_F(data[j,0:5], step)
            out[j,-2] = __D2_Five_Point_Stencil_Assymmetric_B(data[j,-5:], step)
            out[j,-1] = __D2_Five_Point_Stencil_Backward(data[j,-5:], step)
            for i in range(2, data_shape[1]-2):
                out[j,i] = __D2_Five_Point_Stencil(data[j,i-2:i+3], step)

    #######
    if bRotated:
        out = out.T
        
    return out