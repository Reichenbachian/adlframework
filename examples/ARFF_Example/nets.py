import keras
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Flatten, Conv1D, BatchNormalization, MaxPooling1D, Reshape
from keras import regularizers as reg
import attr
import pdb
import keras.backend.tensorflow_backend as K
from adlframework.nets.net import Net


class audio_feature_extractor(Net):
    @Net.build_model_wrapper
    def build_model(self, mode=False):
        PADDING = 'valid'
        model=Sequential()
        model.add(Conv1D(20, 80, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(8))
        model.add(Conv1D(32, 40, strides=2, activation='relu'))
        model.add(MaxPooling1D(8))
        model.add(Conv1D(64, 32, strides=1, activation='relu'))
        model.add(MaxPooling1D(8))
        model.add(Conv1D(128, 16, strides=2, activation='relu'))
        model.add(MaxPooling1D(4))
        model.add(Conv1D(256, 4, strides=1, activation='relu'))
        model.add(Dropout(.5))

        model.add(Activation('softmax', name = 'audio_feature_extractor'))


        print "Model Summary:"
        print model.summary()    
        return model

class dense_network(Net):

    @Net.build_model_wrapper
    def build_model(self):
        model=Sequential()
        model.add(Dense(512, activation='relu', input_shape=self.input_shape))
        model.add(Dense(256, activation='relu', input_shape=self.input_shape))
        model.add(Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(Dense(64, activation='relu', input_shape=self.input_shape))
        model.add(Dense(self.target_shape, activation='relu'))
        model.add(Activation('softmax', name = 'audio_feature_extractor'))


        print "Model Summary:"
        print model.summary()    
        return model
