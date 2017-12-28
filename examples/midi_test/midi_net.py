from keras.models import Sequential
from keras.layers import Dense, LSTM
from adlframework.nets.net import Net


class midi_test(Net):
	@Net.build_model_wrapper
	def build_model(self):
		model = Sequential()
		model.add(LSTM(120, activation='relu',
							input_shape=self.input_shape,
							return_sequences=True))
		model.add(LSTM(120, activation='relu', return_sequences=True))
		model.add(LSTM(120, activation='relu'))
		model.add(Dense(self.target_shape))
		return model