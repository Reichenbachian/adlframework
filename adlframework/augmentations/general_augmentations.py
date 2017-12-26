"""
General augmentations that should be applicable to all
input types with solely restrictions on number of dimensions.

Alex Reichenbach
December 7, 2017
"""

import random
import numpy as np

##########################
#####  Augmentations #####
##########################


def identity(de, tup):
    """
    Returns an unmodified version of the sample.
    Required if an unmodified sample is desired.
    """
    data, label = tup
    return data, label


def add_linear_trend(de, tup, scale=3):
    """
    Adds a linear trend to a 1d piece of data.
    The scale represents the scale to the std
    that will be the slope.

    i.e: m = std*scale

    Prerequisites:
    The input is 1d to the data.
    """
    data, label = tup
    assert len(data.shape) == 1, "The input should be one dimesional."
    std = np.std(data)
    m = np.std(data)/len(data)*scale*(1-2*random.random())
    augmented = [z + m*x for z, x in zip(data, range(0, len(data)))]
    return augmented, label


def offset(de, tup, max_offset_scale=1):
    """
    Adds an offset. max_offset_scale is as follows

    offset = random.random * std * scale
    """
    data, label = tup
    data += random.random()*np.std(data)*max_offset_scale
    return data, label


def add_noise(de, tup, scale=0.2):
    """
    Adds random noise of the following kind.

    noise = random*std*scale
    """
    data, label = tup
    std = np.abs(np.std(data))
    data = data.astype('float64')
    # add noise
    data += np.random.normal(loc=0, scale=std*scale, size=data.shape)
    return data.astype('float64'), label


def timeshift(de, tup, max_shift=.2):
    """
    Timeshifts code forward or backwards.

    max_shift is the max percentage of length
    that will shift.

    i.e: [0,1,2,3,4] -> [3,4,0,1,2]
    """
    data, label = tup
    # shift in time forwards or backwards
    timeshift_fac = max_shift * 2 * \
        (np.random.uniform() - 0.5)  # up to 20% of length
    start = int(de.length * timeshift_fac)
    if (start > 0):
        data = np.pad(data, (start, 0), mode='constant')[0:data.shape[0]]
    else:
        data = np.pad(data, (0, -start), mode='constant')[0:data.shape[0]]
    return data.astype('int16'), label
    