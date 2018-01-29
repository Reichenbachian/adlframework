'''
Why are there three Audio Classes?
One is a base class to prevent redundancy.
There is a fundamental difference between the other two.

In the colloquium, there are two ways to store audio
files: in short segments with one-hot labels. The other
is in larger audio recordings with timestamps and multiple

'''
import attr
import pandas as pd
import scipy.io.wavfile as wav
from adlframework._dataentity import DataEntity
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AudioFileDataEntity(DataEntity):
    '''
    A base class for audio entities.
    '''

    # Sampling rate
    fs   = attr.ib(validator=attr.validators.instance_of(int))

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
        self.length = len(audio)/float(self.fs)
        return audio

class AudioSegmentDataEntity(AudioFileDataEntity):
    '''
    This represents a short audio segment. One that can be clipped on both
    ends for training without losing a significant amount of data. Usually,
    this segment possesses a single label.

    length - length in seconds of audio file
    fs     - sampling rate of audio file
    '''
    # original audio file length
    length   = attr.ib(validator=attr.validators.instance_of(float))

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

    Preconditions
    Labels contain a 'timestamp' column!

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

    '''
    # To-Do: Implement!
    def __init__(self, window_size=None, padding='zeros', sampling_method='linear_interpolation',
                    **kwargs):
        '''
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
        self.padding = padding
        self.indexed = False
        self.sampling_method = sampling_method
        self.window_size = window_size

    def index(self):
        '''
        Sorts the labels by timestamps. Delayed until sampled.
        '''
        assert 'timestamp' in self.labels.columns, "Please use a timestamp column if using an AudioRecordingDataEntity"
        self.labels = self.labels.sort_values(by='timestamp')
        self.indexed = True

        ### To-Do: remove those labels that are outside the range of start_time+window_size*fs

    def discrete_sample(self):
        '''
        Samples without crossing over between labels.
        Meaning the dataframe below will only ever return
        two samples. Sample_1=(2.3, 2.3+window_size*fs) and Sample_2=(4.5, 4.5+window_size*fs)

        | timestamp | metric_1 | metric_1 |
        -----------------------------------
        |    2.3    |    4.2   |   3.5    |
        -----------------------------------
        |    4.2    |    5.2   |   3.4    |
        -----------------------------------
        |    5.2    |    5.4   |   3.1    |
        -----------------------------------
        '''
        # Choose a label, except not the last one, as we can't sample off the end of the file.
        sampled_label = self.labels.iloc[np.random.choice(len(self.labels))] 


    def interpolate_sample(self):
        '''
        Samples continuously. Crossing over labels if necessary.
        For instance, one sample might be (4.4, 4.4+window_size*fs)
        and the label for metric_1 will become the weighted sum.

        | timestamp | metric_1 | metric_1 |
        -----------------------------------
        |    2.3    |    4.2   |   3.5    |
        -----------------------------------
        |    4.2    |    5.2   |   3.4    |
        -----------------------------------
        |    5.2    |    5.4   |   3.1    |
        -----------------------------------
        '''
        
    def get_sample(self):
        '''
        Returns a sample
        '''
        if self.labels is None: # Read labels into memory.
            self.labels = self.retrieval.get_label(self.unique_id)
        if self.indexed:
            self.index()
        if self.data is None: # Read data into memory.
            self.get_data()

        sampled_data = None
        sampled_label = None
        if self.sampling_method == 'linear_interpolation':
            sampled_data, sampled_label  = self.interpolate_sample()
        else if self.sampling_method == 'discrete':
            sampled_data, sampled_label = self.discrete()
        else:
            raise Exception('Sampling method: 'str(self.sampling_method) + ' is not implemented. Check the docstring.')

        return sampled_data, sampled_label
