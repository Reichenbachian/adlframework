from adlframework._dataentity import DataEntity
import arff
import numpy as np
from adlframework.utils import get_logger

logger = get_logger()

class ARFFDataEntity(DataEntity):
    '''
    Represents a arff file.
    '''

    def _read_file(self, id_):
        """
        Receives a file path from retrieval and loads/returns data.
        """
        f = self.retrieval.get_data(id_)
        return np.array([x for x in arff.load(f)])

    def get_sample(self, id_):
        '''
            Given a numpy array of returned sample, it returns a sample.
        '''
        label = self.retrieval.get_label(id_).iloc[0]
        data = self.get_data(id_)
        return data, label
        
