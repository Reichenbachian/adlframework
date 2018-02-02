from keras.layers import Activation, Reshape
from adlframework.utils import get_logger

logger = get_logger()

class Net(object):
	def __init__(self, **kwargs):
		self.softmax = False
		self.PADDING = 'valid'
		self.REGULARIZATION = .001
		self.transfer = False
		self.target_shape = None
		self.input_shape = None
		for key in kwargs:
			setattr(self, key, kwargs[key])

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

			# logger.debug("Input shape to network is ", self.input_shape)
			if self.softmax:
				model.add(Activation('softmax'))
			return model
		return model_wrapped