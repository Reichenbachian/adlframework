from adlframework._dataentity import DataEntity
from madmom.utils.midi import MIDIFile
from madmom.utils import suppress_warnings
from adlframework.utils import get_logger

logger = get_logger()

class MidiDataEntity(DataEntity):
    '''
    Represents a Midi data entitity.
    '''
    
    @suppress_warnings
    def _read_file(self, id_):
        """
        Receives a file path from retrieval and processes it into DataEntity.
        """
        f = self.retrieval.get_data(self.unique_id)
        if self.backend == 'default':
            return converter.parse(f)
        elif self.backend == 'madmom':
            return MIDIFile.from_file(f)
        else:
            raise NotImplemented(str(self.backend)+' is not implemented as a backend')

    def get_sample(self, id_):
        '''
        Returns a midi sample
        '''
        labels = self.retrieval.get_label(id_)
        data = self.get_data()
        return data, None if labels is None else labels.iloc[0]

