import attr
import pandas as pd
import scipy.io.wavfile as wav
from adlframework._dataentity import DataEntity
from adlframework.utils import get_logger
import random
import numpy as np

logger = get_logger()

class AudioFileDataEntity(DataEntity):
    # original audio file length
    length   = attr.ib(validator=attr.validators.instance_of(float))
    # Sampling rate
    fs   = attr.ib(validator=attr.validators.instance_of(int))
    

    def __init__(self, unique_id, retrieval, window_length, pad="zeros"):
        '''
        Window length is in seconds.
        '''
        self.unique_id = unique_id
        self.retrieval = retrieval
        self.window_length = window_length
        self.pad = pad
        self.labels = pd.read_csv(retrieval.get_label(unique_id))

        #### Make sure timestamp column is present. 
        if self.TIMESTAMP_KEY not in self.labels.columns:
            if len(self.labels) == 1:
                self.labels[0] = 0
            else:
                raise Exception('Multiple entries in label without timestamps')

        self.labels = self.labels.sort_values(self.TIMESTAMP_KEY)
        
    def read_file(self):
        """
        Reads the raw data from the retrieval virtual or real file.
        Don't call these
        """
        f = self.retrieval.get_data(self.unique_id)
        # Load wav files
        self.fs, audio = wav.read(f)
        self.length = len(audio)/float(self.fs)
        return audio

    def get_sample(self):
        '''
            Given a numpy array of returned sample, it returns a sample.
        '''
        raw_arr = self.get_data()

        #### Data
        label = self.labels.iloc[int(random.random()*len(self.labels))]
        start_time = label[self.TIMESTAMP_KEY]
        start_index = int(start_time*self.fs)
        raw_arr = raw_arr[start_index:start_index+self.window_length*self.fs]

        ## the below lines are for selecting a subsegment if the length is too long
        if raw_arr.shape[0] < int(self.window_length*self.fs):
            #### Data
            ## the below lines are for zero padding if audio_len < win_length
            pad = None
            if self.pad.lower() == 'zeros':
                pad = np.zeros(int(self.window_length*self.fs))
            elif self.pad.lower() == 'noise':
                pad = np.random.noise(int(self.window_length*self.fs))
            pad[:len(raw_arr)] = raw_arr
            raw_arr = pad

        return raw_arr, label
        
