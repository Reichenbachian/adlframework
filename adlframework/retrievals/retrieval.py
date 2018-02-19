from abc import ABCMeta, abstractmethod
import os

class Retrieval():
    '''
    A retrieval maps an id to a piece of data.
    There should be at most one retrieval per data source.
    '''
    __metaclass__ = ABCMeta
    return_types = ["np array", "file path", "raw"]
    def __init__(self, retrieval_name=None):
        '''
        Init is optionally called by subclasses. It should be called
        if caching is wished to be enabled.
        '''
        assert hasattr(self, '_return_type'), '_return_type must be defined in retrieval in %s' % self.__class__
        if retrieval_name is None:
            print("Retrieval not named, so won't be cached.")
        else:
            f_name = retrieval_name+'.pickle'
            if os.exists(f_name):
                print("WARNING! READING FROM CACHE.")
                pk = pickle.loads(f_name)
                self.load_picklable(f_name)
            else:
                print("Caching ", str(retrieval), " as a pickle file")
                pk = self.get_picklable()
                pickle.dump(pk, retrieval_name+'.pickle')


    @abstractmethod
    def get_data(self, id):
        '''
        Returns a file path or a numpy array as read from the source.
        '''
        raise NotImplementedError("No implementation for read_data provided by class %s" % self.__class__)

    @abstractmethod
    def get_label(self, id):
        '''
        Returns a pandas dataframe.
        '''
        raise NotImplementedError("No implementation for read_label provided by class %s" % self.__class__)

    @abstractmethod
    def list(self, subclassification=""):
        '''
        Returns a list of unique string identifiers.
        '''
        raise NotImplementedError("No implementation for list provided by class %s" % self.__class__)

    def get_picklable(self):
        '''
        Returns a picklable object
        '''
        raise NotImplementedError("No implementation for get_picklable provided by class %s" % self.__class__)

    def load_picklable(self, pickle):
        '''
        Given a saved picklable object, load it.
        '''
        raise NotImplementedError("No implementation for load_picklable provided by class %s" % self.__class__)

    def is_cached(self):
        ### To-Do: Implement Caching
        return False

    def cache(self):
        return
        raise NotImplementedError("Caching has not yet been implemented")

    def return_type(self):
        '''
        Returns what type of data it will return. It must be one of...
        "file path"
        "np array"

        '''
        assert self._return_type in self.return_types, 'Return type is not currently supported.'
        return self._return_type
