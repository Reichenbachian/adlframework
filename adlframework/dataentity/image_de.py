from adlframework._dataentity import DataEntity
from adlframework.utils import get_logger

logger = get_logger()

class ImageFileDataEntity(DataEntity):
    '''
    Represents a single image.
    '''

    def _read_np(self):
        """
        Receives a numpy array from retrieval and processes it into DataEntity's
        format.
        """
        if self.data is None:
            self.data = self.retrieval.get_data(self.unique_id)
        return self.data

    def _read_file(self):
        """
        Receives a file path from retrieval and processes it into DataEntity.
        Recommended to call read_raw to prevent rudundancy.
        """
        raise NotImplemented('read_file is not implemented for ImageFileDataEntity')

    def get_sample(self):
        '''
            Given a numpy array of returned sample, it returns a sample.
        '''
        if self.labels is None: # Read labels into memory.
            self.labels = self.retrieval.get_label(self.unique_id)
        if self.data is None: # Read data into memory.
            self.get_data()
        return self.data, self.labels.iloc[0]
        
