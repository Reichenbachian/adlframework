from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from adlframework.nets.net import Net

class mnist_net(Net):
	@Net.build_model_wrapper
	def build_model(self):
			model = Sequential()
			model.add(Conv2D(32, kernel_size=(3, 3),
												   padding=self.PADDING,
												   activation='relu',
												   input_shape=self.input_shape))
			model.add(Conv2D(64, (3, 3), activation='relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.25))
			model.add(Flatten())
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(self.target_shape, activation='softmax'))
			return model

