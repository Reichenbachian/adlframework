from keras.layers import Activation, Reshape
import attr
import logging

logger = logging.getLogger(__name__)

@attr.s
class Net(object):
	target_shape = attr.ib()
	input_shape = attr.ib()
	transfer = attr.ib(default=False)
	softmax = attr.ib(default=False)
	PADDING = attr.ib(default='valid')
	REGULARIZATION = attr.ib(default=0.001)

	@staticmethod
	def build_model_wrapper(build_model):
		def model_wrapped(self):
			### Allows tuple
			reshape_out_shape = None
			if type(self.target_shape) is tuple:
				reshape_out_shape = self.target_shape
				t = 1
				for i in self.target_shape:
					t *= i
				self.target_shape = t

			model = build_model(self) # Build the model

			if reshape_out_shape != None: # Add reshape layer to reshape
				model.add(Reshape(reshape_out_shape))

			logger.log(logging.INFO, "Input shape to network is ", model.input_shape)
			if self.softmax:
				model.add(Activation('softmax'))
			return model
		return model_wrapped