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
    # Sampling rate
    fs   = attr.ib(validator=attr.validators.instance_of(int))

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
    Labels contain a Timestamps column!

    length - length in seconds of audio file
    fs     - sampling rate of audio file
    '''
    # To-Do: Implement!
    def __init__(self, padding, **kwargs):
        DataEntity.__init__(self, **kwargs)
        self.padding = padding