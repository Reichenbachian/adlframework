import keras
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import *
from keras import regularizers as reg
from adlframework.nets.net import Net
import attr

@attr.s
class medium_model(Net):
	target_shape = attr.ib()
	input_shape = attr.ib()
	transfer = attr.ib(default=False)
	PADDING = attr.ib(default='valid')
	softmax = attr.ib(default=False)

	def build_model(self):
		print "Input shape prior to processing via 1st conv layer: ", self.input_shape
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding=self.PADDING,
		                 input_shape=self.input_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# model.add(Conv2D(64, (3, 3), padding=self.PADDING))
		# model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.target_shape))
		if self.softmax:
			model.add(Activation('softmax'))
		return model