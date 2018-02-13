from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv1D, LSTM, MaxPooling1D, PADDING, model
from adlframework.nets.net import Net
from keras import regularizers as reg

class planet_structure_lstm(Net):
    @Net.build_model_wrapper
    def build_model(self, mode=False):
        model = Sequential()
        model.add(LSTM(32, name="lstm_1", input_shape=self.input_shape, return_sequences=True))
        model.add(LSTM(64, name="lstm_2", return_sequences=True))
        model.add(LSTM(128, name="lstm_3", return_sequences=True))
        model.add(LSTM(2, name = 'planet_lstm_output'))
        model.add(Activation('softmax', name = 'planet_other_activation'))

        return model

class planet_structure_lrcn(Net):
    @Net.build_model_wrapper
    def build_model(self, mode=False):
        model.add(Conv1D(32, 64, strides=4, padding=PADDING, input_shape=self.input_shape, trainable=not self.transfer,
                         kernel_regularizer = reg.l2(self.REGULARIZATION)))
        model.add(BatchNormalization(trainable=not self.transfer))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=8))
        # 1 Kernel sees : 128 * 8 * 4 + 256 + 256 of field 
        model.add(Conv1D(32, 16, strides=2, padding=PADDING, trainable=not self.transfer,
                         kernel_regularizer = reg.l2(self.REGULARIZATION)))
        model.add(BatchNormalization(trainable=not self.transfer))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(LSTM(32, name="32_dense", return_sequences=True))
        model.add(LSTM(16, name="16_dense", return_sequences=True))
        model.add(LSTM(self.num_classes, name = 'planet_lrcn_output'))
        model.add(Activation('softmax', name = 'planet_other_activation'))

        return model

class truncated_planet_structure_lrcn(Net):
    @Net.build_model_wrapper
    def build_model(self, mode=False):
        model = Sequential()
        model.add(Conv1D(32, 64, strides=4, padding=PADDING, input_shape=self.input_shape, trainable=not self.transfer,
                         kernel_regularizer = reg.l2(self.REGULARIZATION)))
        model.add(BatchNormalization(trainable=not self.transfer))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=8))
        model.add(LSTM(32, name="32_dense", return_sequences=True))
        model.add(LSTM(self.num_classes, name = 'planet_lrcn_output'))
        model.add(Activation('softmax', name = 'planet_other_activation'))

        return model

class lrcn_v3(Net):
    @Net.build_model_wrapper
    def build_model(self, mode=False):
        model = Sequential()
        model.add(Conv1D(32, 64, strides=4, padding=PADDING, input_shape=self.input_shape, trainable=not self.transfer,
                         kernel_regularizer = reg.l2(self.REGULARIZATION)))
        model.add(Conv1D(64, 16, strides=4, padding=PADDING, input_shape=self.input_shape, trainable=not self.transfer,
                         kernel_regularizer = reg.l2(self.REGULARIZATION)))
        model.add(BatchNormalization(trainable=not self.transfer))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=8))
        model.add(LSTM(32, name="32_dense", return_sequences=True))
        model.add(LSTM(self.num_classes, name = 'planet_lrcn_output'))
        model.add(Activation('softmax', name = 'planet_other_activation'))

        return model
