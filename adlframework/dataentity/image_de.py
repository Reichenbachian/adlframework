from adlframework._dataentity import DataEntity
from adlframework.utils import get_logger

logger = get_logger()

class ImageFileDataEntity(DataEntity):
    '''
    Represents a single image.
    '''

    def _read_np(self, id_):
        """
        Receives a numpy array from retrieval and processes it into DataEntity's
        format.
        """
        return self.retrieval.get_data(id_)

    def _read_file(self, id_):
        """
        Receives a file path from retrieval and processes it into DataEntity.
        Recommended to call read_raw to prevent rudundancy.
        """
        raise NotImplemented('read_file is not implemented for ImageFileDataEntity')

    def get_sample(self, id_):
        '''
            Given a numpy array of returned sample, it returns a sample.
        '''
        return self.get_data(id_), self.retrieval.get_label(id_)
        
