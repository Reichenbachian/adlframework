import pdb
import numpy as np
######################
####   FILTERS  ######
######################
def min_array_shape(sample, min_shape=None):
    '''
    Checks if dimensions meet minimum in all directions.
    If dimensions don't match up, returns false.
    '''
    data, _ = sample
    for i in range(len(min_shape)):
        try:
            if min_shape[i] > len(sample[i]):
                return False
        except IndexError:
            return False
    return True

def ignore_label(sample, labelnames):
    """ Reject samples for which the value for any of labelnames is 1, accept otherwise"""
    return threshold_label(sample, labelnames, 1, keep=False)

def accept_label(sample, labelnames):
    """ Only accept samples which has the value of 1 for one or more of the given labelnames"""
    return threshold_label(sample, labelnames, 1)

def threshold_label(sample, labelnames, threshold,
                    greater_than=True, keep_unpresent=False,
                    keep=True, equal_to=True):
    """
    If keep is True:
        The entity will keep those values
        that are (greater_than or not greater_than)
        the threshold value.
    If keep is False:
        The entity will discard those values
        that are (greater_than or not greater_than)
        the threshold value.
    """
    data, label = sample
    print label
    if isinstance(labelnames, basestring):
        labelnames = [labelnames]
    for labelname in labelnames:
        if labelname not in label: # Remove unpresent
            return keep_unpresent
        if (greater_than and label[labelname] >= threshold) or \
            (not greater_than and label[labelname] <= threshold) or \
            (equal_to and label[labelname] == threshold):
            return keep
    return not keep
