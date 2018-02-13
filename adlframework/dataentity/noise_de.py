from adlframework._dataentity import DataEntity
import numpy as np
from adlframework.utils import get_logger

logger = get_logger()

class NoiseDataEntity(DataEntity):
    
    def __init__(self, *args, **kwargs):
        self.shape = kwargs.pop('shape')
        assert self.shape != None, "Please specify shape for noise data entity"
        super(NoiseDataEntity, self).__init__(*args, **kwargs)

    def _read_raw(self, id_):
        """
        Returns random noise of type.
        """
        try:
            return np.random.uniform(size=self.shape)
        except KeyError:
            raise Exception('Make sure keys "low" and "high are defined in noise_metadata."')

    def get_sample(self, id_):
        raw_arr = self._read_raw(id_)
        return raw_arr, None
