from music21 import converter
from adlframework.dataentity.dataentity import DataEntity
import logging

logger = logging.getLogger(__name__)

class MidiDataEntity(DataEntity):
    '''
    Represents a Midi data entitity.
    '''
    
    def __init__(self, unique_id, retrieval, sampler=None):
        self.unique_id = unique_id
        self.retrieval = retrieval
        self.sampler = sampler
        self.labels = None
        self.data = None

    def _read_raw(self):
        """
        Read from raw midi data.
        """
        raise NotImplemented('Reading from raw midi data is not implemented.')

    def _read_file(self):
        """
        Receives a file path from retrieval and processes it into DataEntity.
        """
        f = self.retrieval.get_data(self.unique_id)
        mid = converter.parse(f)
        return mid

    def get_sample(self):
        '''
            Returns a midi sample
        '''
        if self.labels is None: # Read labels into memory.
            self.labels = self.retrieval.get_label(self.unique_id)
        if self.data is None: # Read data into memory.
            self.get_data()
        return self.data, None if self.labels is None else self.labels.iloc[0]

