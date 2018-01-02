## Nets
A *_nets.py file denotes a file that contains a specific neural network. It should follow the following syntax...

```
from adlframework.nets.net import Net
class a_network(Net):
	@Net.build_model_wrapper
	def build_model(self):
	...
	return model
```

Every net contains the following variables that are set during variable creation
 - input_shape: a tuple representing the input shape of the model.
 - target_shape: a tuple representing the target shape of the model.
 - softmax: a boolean value that adds a softmax layer to the end of a model. (default=False)
 - PADDING: a boolean value that adds a convolutional padding type. (default='valid')
 - REGULARIZATION: a float value that represents the regularization constant
 - transfer: a boolean value that represents whether transfer learning should happen.
 - all other keyword variables are set to `self.[variable name]=value`