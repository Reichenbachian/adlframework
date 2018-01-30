'''
Why are there three Audio Classes?
One is a base class to prevent redundancy.
There is a fundamental difference between the other two.

In the colloquium, there are two ways to store audio
files: in short segments with one-hot labels. The other
is in larger audio recordings with timestamps and multiple

'''
import pandas as pd
import scipy.io.wavfile as wav
from adlframework._dataentity import DataEntity
import random
import numpy as np
import logging
import pdb

logger = logging.getLogger(__name__)

class AudioFileDataEntity(DataEntity):
    '''
    A base class for audio entities.

    length - length in seconds of audio file
    fs     - sampling rate of audio file
    '''

    def __init__(self, **kwargs):
        DataEntity.__init__(self, **kwargs)
        self.fs = None
        self.length = None

    def get_sample(self):
        '''
        Should never be called
        '''
        raise Exception("AudioFileDataEntity is deprecated. Use AudioSegmentDataEntity or AudioRecordingDataEntity.")
    
    def _read_file(self):
        '''
        Reads the raw data from the retrieval virtual or real file.
        Don't call these
        '''
        f = self.retrieval.get_data(self.unique_id)
        self.fs, self.data = wav.read(f)
        self.length = len(self.data)/float(self.fs)
        return self.data

class AudioSegmentDataEntity(AudioFileDataEntity):
    '''
    This represents a short audio segment. One that can be clipped on both
    ends for training without losing a significant amount of data. Usually,
    this segment possesses a single label.
    '''

    def get_sample(self):
        '''
        Returns a sample
        '''
        if self.labels is None: # Read labels into memory.
            self.labels = self.retrieval.get_label(self.unique_id)
        if self.data is None: # Read data into memory.
            self.get_data()

        return self.data, self.labels
        
class AudioRecordingDataEntity(AudioFileDataEntity):
    '''
    This represents a longer audio segment. One that has many internal
    labels delimited by timestamps.

    Arguments
    window_size: The size of the window in 

    Preconditions
    Labels contain a 'timestamp' column!

    -----------------------------------
    | timestamp | metric_1 | metric_1 |
    -----------------------------------
    |    2.3    |    4.2   |   3.5    |
    -----------------------------------
    |    4.2    |    5.2   |   3.4    |
    -----------------------------------
    ...etc...

    It is assumed, in the example above that 4.2 is metric_1 for the 
    time between 2.3 and 4.2.

    Class Variables
    ---------------
    length       - length in seconds of audio file
    fs           - sampling rate of audio file
    window_size  - sampling rate of audio file
    self.data    - a 1-d representation of audio.

    '''
    # To-Do: Implement!
    def __init__(self, window_size=None, padding='zeros',
                    timestamp_units='seconds', window_units='seconds',
                    sampling_method='discrete', timestamp_column='timestamp',
                    **kwargs):
        '''
        TO-DO: Implement padding
        TO-DO: Implement filtering

        window_size: In seconds. If it is None, variable lengths will be returned.
        padding
            - 'zeros': pad with zeros
            - 'noise': pad with noise
        sampling_method(For information check class docstring):
            - 'linear_interpolation': combine i and i+1 label, if both are available(i.e. i+1 hasn't been removed by filter),
                                      and randomly sample between them.
            - 'discrete': Don't cross over between labels, and keep labels untouched.
        '''
        DataEntity.__init__(self, **kwargs)
        assert window_units == 'seconds' or window_units == 'frames', 'window_size may only be seconds or frames'
        assert timestamp_units == 'seconds' or window_units == 'frames', 'timestamp_units may only be seconds or frames'
        assert not (window_size == None and sampling_method == 'linear_interpolation'), 'If it is linear interpolating, a window_size must be provided.'
        assert sampling_method == 'discrete' or sampling_method == 'linear_interpolation', 'Only discrete or linear_interpolation are implemented'
        self.padding = padding
        self.indexed = False
        self.sampling_method = sampling_method
        self.window_size = window_size
        self.timestamp_units = timestamp_units
        self.window_units = window_units
        self.tc = timestamp_column
        self.converted_fs = False

    def index(self):
        '''
        Sorts the labels by timestamps. Delayed until sampled.
        '''
        self.labels = self.labels.sort_values(by=self.tc)
        self.indexed = True

        ### Remove labels that are outside the range of start_time+window_size*fs
        if self.window_size:
            max_time = len(self.data) - self.window_size
            max_time = max_time if self.timestamp_units == 'frames' else max_time/self.fs
            self.labels = self.labels[(self.labels[self.tc] < max_time) & (self.labels[self.tc] > 0)]


    def discrete_sample(self):
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
        # Choose a start time
        label_i = np.random.choice(len(self.labels))
        sampled_label = self.labels.iloc[label_i] 
        start_time = sampled_label[self.tc]
        end_time = None
        if self.timestamp_units == 'seconds': ### Convert to frame
            start_time *= self.fs

        # Create an end time
        end_time = None
        if self.window_size == None:
            if label_i == len(self.labels)-1:
                end_time = len(self.data)
            else:
                end_time = self.labels.iloc[label_i+1]
                if self.timestamp_units == 'seconds': ### Convert to frame
                    end_time *= self.fs
        else:
            end_time = start_time + self.window_size

        # Create and return sample
        return self.data[int(start_time):int(end_time)], sampled_label


    def interpolate_sample(self):
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
        # Get start and end time and data
        start_frame = int(np.random.random()*len(self.data))
        end_frame = start_frame + self.window_size
        sampled_data = self.data[start_frame:end_frame]

        # Create label
        start_timestamp = start_frame if self.timestamp_units == 'frames' else start_frame/self.fs
        end_timestamp = end_frame if self.timestamp_units == 'frames' else end_frame/self.fs
        start_label_i = (self.labels[self.tc]-start_timestamp).abs().argsort()[0]
        end_label_i = (self.labels[self.tc]-end_timestamp).abs().argsort()[0]
        sample_label = self.labels.drop(self.tc, axis=1)[start_label_i:end_label_i].mean() # Average two nearest labels. Also remove timestamps column.

        return sampled_data, sample_label
        
    def get_sample(self):
        '''
        Returns a sample
        '''
        if self.labels is None: # Read labels into memory.
            self.labels = self.retrieval.get_label(self.unique_id)
            assert self.tc in self.labels.columns, "Please specify a timestamp column present in labels if using an AudioRecordingDataEntity"
        if self.data is None: # Read data into memory.
            self.get_data()
            #### Convert second units to frame size
            if self.window_units == 'seconds' and not self.converted_fs:
                self.window_size = int(self.window_size*self.fs)
                self.converted_fs = True # This check is necessary because this isn't always run once. In the case data is tossed for memory preservation.
        if self.indexed:
            self.index()

        sampled_data = None
        sampled_label = None
        if self.sampling_method == 'linear_interpolation':
            sampled_data, sampled_label  = self.interpolate_sample()
        else:
            sampled_data, sampled_label = self.discrete_sample()

        return sampled_data, sampled_label
