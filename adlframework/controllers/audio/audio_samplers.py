import pandas as pd
import scipy.io.wavfile as wav
from adlframework._dataentity import DataEntity
import random
import numpy as np
import pdb
from adlframework.utils import get_logger

logger = get_logger()

def crop_possible_labels(data, labels, fs, label_units, window, tc):
    '''
    Crops the impossibile labels from the labels.
    For instance, labels starting after file ends.
    Or one before, but whose distance from the end
    is less than the window length.

    '''
    if window == None:
        return
    labels = labels[labels[tc] > 0] # Remove all negative labels
    max_time = len(data) / float(fs) if label_units == 'seconds' else len(data)
    window = window / float(fs) if label_units == 'seconds' else window
    labels = labels[labels[tc] < max_time - window]
    return labels


def discrete_sample(sample, sampling_rate=None, timestamp_column='Timestamps', timestamp_units='seconds',
                    window_size=None, window_units='seconds'):
        '''
        Samples without crossing over between labels.
        Meaning the dataframe below will only ever return
        two samples. Sample_1=(2.3, 2.3+window_size) and Sample_2=(4.5, 4.5+window_size)

        If window_size is None, the length of the sample is between
        the timestamp sampled and the next.

        -----------------------------------
        | timestamp | metric_1 | metric_1 |
        -----------------------------------
        |    2.3    |    4.2   |   3.5    |
        -----------------------------------
        |    4.2    |    5.2   |   3.4    |
        -----------------------------------
        |    5.2    |    5.4   |   3.1    |
        -----------------------------------
        '''
        assert timestamp_units == 'seconds' or timestamp_units == 'samples', 'Only samples or seconds are allowed for timestamp_units'
        assert window_units == 'seconds' or window_units == 'samples', 'Only samples or seconds are allowed for window_units'
        assert sampling_rate != None, 'Please define a sampling rate for discrete_sample'
        data, labels = sample

        if window_units == 'seconds' and window_size != None:
            window_size = int(window_size*sampling_rate)

        labels = crop_possible_labels(data, labels, sampling_rate, timestamp_units, window_size, timestamp_column)

        # Choose a start time
        label_i = np.random.choice(len(labels))
        sampled_label = labels.iloc[label_i] 
        start_time = sampled_label[timestamp_column]
        end_time = None
        if timestamp_units == 'seconds': ### Convert to frame
            start_time *= sampling_rate

        # Create an end time
        end_time = None
        if window_size == None:
            if label_i == len(labels)-1:
                end_time = len(data)
            else:
                end_time = labels.iloc[label_i+1][timestamp_column]
                if timestamp_units == 'seconds': ### Convert to frame
                    end_time *= sampling_rate
        else:
            end_time = int(start_time) + window_size
        sampled_data = data[int(start_time):int(end_time)]

        # Create and return sample
        return sampled_data, sampled_label


def interpolate_sample(sample, window_size=None, window_units='seconds',
                        timestamp_column='Timestamps', timestamp_units='seconds'):
    '''
    Samples continuously. Crossing over labels if necessary.
    For instance, one sample might be (4.4, 4.4+window_size*fs)
    and the label for metric_1 will become a weighted sum.

    The weighted sum is computed as follows.
    1) Get two nearest labels to start and end point
    2) Compute mean of all labels between them inclusive

    -----------------------------------
    | timestamp | metric_1 | metric_1 |
    -----------------------------------
    |    2.3    |    4.2   |   3.5    |
    -----------------------------------
    |    4.2    |    5.2   |   3.4    |
    -----------------------------------
    |    5.2    |    5.4   |   3.1    |
    -----------------------------------
    '''
    data, labels = sample
    assert sampling_rate != None, 'Please define a sampling rate for discrete_sample'
    assert timestamp_units == 'seconds' or timestamp_units == 'samples', 'Only samples or seconds are allowed for timestamp_units'
    assert window_units == 'seconds' or window_units == 'samples', 'Only samples or seconds are allowed for window_units'

    if window_units == 'seconds':
        window_size = int(window_size*sampling_rate)

    labels = crop_possible_labels(data, labels, sampling_rate, timestamp_units, window_size, timestamp_column)

    # Get start and end time and data
    start_frame = int(np.random.random()*len(data))
    end_frame = start_frame + window_size
    sampled_data = data[start_frame:end_frame]

    # Create label
    start_timestamp = start_frame if timestamp_units == 'samples' else start_frame/sampling_rate
    end_timestamp = end_frame if timestamp_units == 'samples' else end_frame/sampling_rate
    start_label_i = (labels[timestamp_column]-start_timestamp).abs().argsort().iloc[0]
    end_label_i = (labels[timestamp_column]-end_timestamp).abs().argsort().iloc[0]
    sample_label = labels.drop(timestamp_column, axis=1)[start_label_i:end_label_i].mean() # Average two nearest labels. Also remove timestamps column.

    return sampled_data, sample_label
