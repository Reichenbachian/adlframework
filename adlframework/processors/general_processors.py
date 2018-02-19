import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from keras.utils import to_categorical
import pdb
from adlframework.utils import get_logger

logger = get_logger()

pd.options.mode.chained_assignment = None

#########################
### Label Processors  ###
#########################
def pdb_trace(sample, targets=[], reverse=None):
    '''
    Interupts program at controller location for debugging purposes
    '''
    pdb.set_trace()

def to_np_arr(sample, convert_data=True, convert_label=False):
    '''
    Converts either or the data and the label of the sample to
    a numpy array.
    '''
    data, label = sample
    if convert_data:
        data = np.array(data)
    if convert_label:
        label = np.array(label)
    return data, label

def map_labels(sample, targets=[], reverse=None):
    '''
    Maps a pandas label to an array.

    Reverse should be an array that will subtract the value from 1.
    '''
    assert len(targets) != 0, 'Please specify output labels'
    data, label = sample
    if reverse == None:
        reverse = [False for x in range(len(targets))]
    return data, [1-label[x] if reverse[i] else label[x] for i, x in enumerate(targets)]

def make_categorical(sample, num_classes=None):
    """
    Converts integer array to categorical labels for categorical_crossentropy
    """
    data, label = sample
    return data, to_categorical(label, num_classes=num_classes)[0]

warned_threshold_labels = False
def threshold_labels(sample, threshold=.5, targets=None):
    '''
    Maps continuous labels to binary labels.
    '''
    data, label = sample
    assert targets is not None, 'Please specify targets for threshold_labels. Otherwise it will do nothing.'
    assert type(label) is list or type(label) is pd.Series, 'Label from tup must be a numpy Series or a list'
    if type(label) is list and not warned_threshold_labels:
        warned_threshold_labels = True
        logger.warn('Make sure that targets and label are in same order! Otherwise you will get unintended results.')
    for i in range(len(targets)):
        if type(label) is pd.Series:
            i = targets[i]
        label[i] = int(label[i] > threshold)
    return data, label

def crop(sample, shape):
    '''
    To-Do: Make arbitrary dimensional
    '''
    data, label = sample
    padded = np.zeros(shape)
    padded[:data.shape[0],:data.shape[1]] = data[:padded.shape[0],:padded.shape[1]]
    return padded, label

#########################
#### Data Processors  ###
#########################
def zeros(sample, shape):
    data, label = sample
    assert data.ndim == len(shape), "Data and shape must have same number of dimensions."
    zeros = np.zeros(shape)
    s = [slice(None, x) for x in data.shape]
    zeros[s] = data
    return zeros, label

def reshape(sample, shape, reshape_label=False, reshape_data=True):
    """
    Reshapes the input array to a desired output shape.
    """
    data, label = sample
    if reshape_label:
        label = label.reshape(shape)
    if reshape_data:
        data = data.reshape(shape)
    return data, label

def remove_outliers(sample, threshold=3):
    """
    Removes data points which are threshold times
    further from the IQR, which is the distance from
    the upper and lower quartile.
    """
    data, label = sample
    mean = np.mean(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = (upper_quartile - lower_quartile) * threshold
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    for i in range(len(data)):
        y = data[i]
        if y < quartileSet[0] or y > quartileSet[1]:
            data[i] = mean
    return data, label

def smooth_gaussian_processor(sample, k=2):
    data = sample
    assert len(np.shape(np.array(data))) == 1, 'Please only give smooth_gaussian_processor a one dimensional input.'
    smoothed = gaussian_filter1d(data, k)
    return smoothed

def amplitude_extractor(sample):
    """
    Subtracts the mean and divides by the maximum.
    Makes no change if the array is solely comprised of 0s.
    """
    data,label = sample
    data -= np.mean(data)
    max_ = max(np.abs(data))
    if max_ == 0:
        return data, label
    return data/max_, label

def subtract_gaussian_filter(sample, k=10):
    """
    Subtract the gaussian filter of the array.
    """
    data, label = sample
    smoothed = data.copy()
    smoothed = gaussian_filter1d(smoothed, k)
    data -= smoothed
    return data, label
