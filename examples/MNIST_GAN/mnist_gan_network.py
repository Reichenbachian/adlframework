from keras.models import Sequential, Model
from keras.layers import *
from keras import initializers
from adlframework.nets.net import Net
from keras.optimizers import Adam

class mnist_gan_network(Net):
	@Net.build_model_wrapper
	def build_model(self):
		adam = Adam(lr=0.0002, beta_1=0.5)
		generator = Sequential()
		generator.add(Dense(256, input_dim=self.randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		generator.add(LeakyReLU(0.2))
		generator.add(Dense(512))
		generator.add(LeakyReLU(0.2))
		generator.add(Dense(1024))
		generator.add(LeakyReLU(0.2))
		generator.add(Dense(784, activation='tanh'))
		generator.compile(loss='binary_crossentropy', optimizer=adam)

		discriminator = Sequential()
		discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		discriminator.add(LeakyReLU(0.2))
		discriminator.add(Dropout(0.3))
		discriminator.add(Dense(512))
		discriminator.add(LeakyReLU(0.2))
		discriminator.add(Dropout(0.3))
		discriminator.add(Dense(256))
		discriminator.add(LeakyReLU(0.2))
		discriminator.add(Dropout(0.3))
		discriminator.add(Dense(1, activation='sigmoid'))
		discriminator.compile(loss='binary_crossentropy', optimizer=adam)

		self.discriminator = discriminator
		self.generator = generator

		# Combined network
		discriminator.trainable = False
		ganInput = Input(shape=(self.randomDim,))
		x = generator(ganInput)
		ganOutput = discriminator(x)
		gan = Model(inputs=ganInput, outputs=ganOutput)
		self.gan = gan
		return gan