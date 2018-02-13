from keras.models import Sequential
from keras.layers import *
from adlframework.nets.net import Net


class large_model(Net):
	@Net.build_model_wrapper
	def build_model(self, mode=False):
		model = Sequential()
		REGULARIZATION = 0.0005
		model.add(Conv1D(16, 64, strides=2, padding=self.PADDING, input_shape=self.input_shape, trainable=not self.transfer,
						 kernel_regularizer = reg.l2(self.REGULARIZATION)))
		model.add(BatchNormalization(trainable=not self.transfer))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=8))
		model.add(Conv1D(32, 32, strides=2, padding=self.PADDING, trainable=not self.transfer,
						 kernel_regularizer = reg.l2(self.REGULARIZATION)))
		model.add(BatchNormalization(trainable=not self.transfer))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=8))
		model.add(Conv1D(64, 16, strides=2, padding=self.PADDING,
						 kernel_regularizer = reg.l2(self.REGULARIZATION)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv1D(128, 8, strides=2, padding=self.PADDING,
						 kernel_regularizer = reg.l2(self.REGULARIZATION)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(64, name="64_dense", kernel_regularizer = reg.l2(REGULARIZATION)))
		model.add(Dropout(rate=0.5))
		model.add(Dense(self.num_classes, name = 'audio_cnn_output'))

		return model

class medium_model(Net):
	@Net.build_model_wrapper
	def build_model(self, mode=False):
		model = Sequential()
		model.add(Conv1D(32, 64, strides=2, padding='same', input_shape=self.input_shape, trainable=not self.transfer))
		model.add(BatchNormalization(trainable=not self.transfer))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=8))
		model.add(Conv1D(64, 32, strides=2, padding='same', trainable=not self.transfer))
		model.add(BatchNormalization(trainable=not self.transfer))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=8))

		model.add(Conv1D(128, 16, strides=2, padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		model.add(Conv1D(256, 8, strides=2, padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		model.add(Flatten())
		model.add(Dense(128))
		model.add(Dropout(rate=0.5))

		model.add(Dense(self.num_classes, name = 'audio_cnn_output'))
		return model

class small_model(Net):
	@Net.build_model_wrapper
	def build_model(self, mode=False):
		model = Sequential()

		model.add(Flatten(input_shape=self.input_shape))
		model.add(Dense(512))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))

		model.add(Dense(self.num_classes))

		return model
