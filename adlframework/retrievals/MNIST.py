from adlframework.retrievals.retrieval import Retrieval
from sklearn.datasets import fetch_mldata
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MNIST_retrieval(Retrieval):
    '''
    Get access to the mnist data
    '''
    _train_image_location = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    _label_image_location = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    _return_type = 'np array'
    def __init__(self, mnist_data_location='./'):
        super(MNIST_retrieval, self).__init__()
        logger.info('Downloading mnist data...')
        self.mnist = fetch_mldata('MNIST original', data_home=mnist_data_location)
        logger.info('Done downloading mnist data...')

    def get_data(self, id_):
        '''
        Returns mnist image
        '''
        return np.array(self.mnist.data[id_])

    def get_label(self, id_):
        '''
        Returns a label
        '''
        return pd.DataFrame({0: [self.mnist.target[id_]]})

    def list(self):
        return range(len(self.mnist.data))
