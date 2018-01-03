from adlframework._dataentity import DataEntity
import numpy as np
import logging



logger = logging.getLogger(__name__)

class NoiseDataEntity(DataEntity):
    
    def __init__(self, id_, shape, **noise_metadata):
        self.unique_id = id_
        self.shape = shape
        self.noise_metadata = noise_metadata

    def read_raw(self):
        """
        Returns random noise of type.
        """
        try:
            return np.random.uniform(size=self.shape, **self.noise_metadata)
        except KeyError:
            raise Exception('Make sure keys "low" and "high are defined in noise_metadata."')

    def get_sample(self):
        raw_arr = self.read_raw()
        return raw_arr, None
