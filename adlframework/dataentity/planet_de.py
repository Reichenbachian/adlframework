import pandas as pd
from adlframework._dataentity import DataEntity
import numpy as np
import idlsave
from adlframework.utils import get_logger

logger = get_logger()

class PlanetDataEntity(DataEntity):
    
    def __init__(self, pad="zeros", *args, **kwargs):
        '''
        window_length is in indices
        '''
        self.shape = kwargs.pop('window_length')
        self.label = pd.read_csv(retrieval.read_label(unique_id)).iloc[0]

    def _read_raw(self):
        """
        Reads the raw data from the retrieval virtual or real file
        """
        f = self.retrieval.read_data(self.unique_id)
        # Load wav files
        k = idlsave.read(f, verbose=False)['k']
        self.flux = k.f[0]
        self.timestamps = k.t[0]
        self.length = len(self.flux)
        return flux

    def get_sample(self):
        raw_arr = self.read_raw()
        if raw_arr.shape[0] < int(self.window_length*self.fs):
            ## If the data is too short
            pad = None
            if self.pad.lower() == 'zeros':
                pad = np.zeros(int(self.window_length*self.fs))
            elif self.pad.lower() == 'noise':
                pad = np.random.noise(int(self.window_length*self.fs))
            pad[:len(raw_arr)] = raw_arr
            return pad, self.label
        else:
            ### If the data is too long
            start_time = raw_arr.shape[0] - self.window_length
            data = raw_arr[start_time:start_time+self.window_length]
            return data, self.label
        
    def get_bls_data(self):
        try: 
            bls = idlsave.read(self.get_cached_bls_filename(), verbose=False)['blsstr']
            self.snr = [x.snr for x in bls]
            self.snr_periods = [x.period for x in bls][0]
            self.period = [x.pnew[0] for x in bls][0]
            self.duration = [x.pnew[2] for x in bls][0]
            self.t0 = [x.pnew[1] for x in bls][0]
            self.impact_parameter = [x.pnew[3] for x in bls][0]
            self.limb_darkening = [x.pnew[4] for x in bls][0]
            self.radius_ratio = [x.pnew[6] for x in bls][0]
        except Exception as e:
            logger.warn("Failed to read downloaded file :" + self.get_cached_search_filename()+ ' ' +str(e))
        return [self.period, self.t0, self.impact_parameter, self.limb_darkening, None, self.radius_ratio]

