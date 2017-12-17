from functools import partial
### Data
from adlframework.retrievals.MNIST import MNIST_retrieval
from adlframework.datasource import DataSource
from adlframework.dataentity.image_de import ImageFileDataEntity
### Model
from adlframework.nets.image_nets import medium_model
from adlframework.experiment import Experiment
### Controllers
from adlframework.processors.general_processors import reshape, make_categorical

### Controllers
processors = [partial(reshape, out_shape=(28, 28, 1)),
			  partial(make_categorical, num_classes=10)]

### Load Data
mnist_retrieval = MNIST_retrieval()
mnist_ds = DataSource(mnist_retrieval, ImageFileDataEntity, processors=processors)

### Load network
net = medium_model(input_shape=(28, 28, 1), target_shape=10)

### Create and run experiment
exp = Experiment(train_datasource=mnist_ds,
					network=net)
exp.run()