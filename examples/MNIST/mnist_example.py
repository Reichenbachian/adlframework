from functools import partial
### Data
from adlframework.retrievals.MNIST import MNIST_retrieval
from adlframework.datasource import DataSource
from adlframework.dataentity.image_de import ImageFileDataEntity
### Model
from mnist_test import mnist_test
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from adlframework.experiment import Experiment
### Controllers
from adlframework.processors.general_processors import reshape, make_categorical
### Callbacks
from adlframework.callbacks.image_callbacks import SaveValImages

import pdb


### Controllers
controllers = [partial(reshape, out_shape=(28, 28, 1)),
			  partial(make_categorical, num_classes=10)]

### Load Data
mnist_retrieval = MNIST_retrieval()
mnist_ds = DataSource(mnist_retrieval, ImageFileDataEntity, controllers=controllers)

train_ds, temp = DataSource.split(mnist_ds, split_percent=.6)
val_ds, test_ds = DataSource.split(temp, split_percent=.6)

### Load network
net = mnist_test(input_shape=(28, 28, 1), target_shape=10)

### Callbacks
callbacks = []

### Create and run experiment
exp = Experiment(train_datasource=train_ds,
					validation_datasource=val_ds,
					test_datasource=test_ds,
					network=net,
					metrics=['mae', 'acc'],
					loss=categorical_crossentropy,
					optimizer=Adadelta(),
					label_names=list(range(10)),
					callbacks=callbacks,
					workers=3,
					epochs=10)
exp.run()