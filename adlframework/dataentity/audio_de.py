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
import pdb
from adlframework.utils import get_logger

logger = get_logger()

class AudioSegmentDataEntity(DataEntity):
    '''
    An attempt at a general audio entity.

    If it fails, check git history for previous versions.
    '''

    def get_sample(self, id_):
        '''
        Returns a sample
        '''
        labels = self.retrieval.get_label(id_)
        data = self.get_data(id_)

        return data, labels
    
    def _read_file(self, id_):
        '''
        Reads the raw data from the retrieval virtual or real file.
        Don't call these
        '''
        f = self.retrieval.get_data(id_)
        self.fs, data = wav.read(f)
        return data
