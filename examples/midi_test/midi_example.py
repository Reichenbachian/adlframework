from functools import partial
### Data
from adlframework.retrievals.GlobSearch import GlobSearch
from adlframework.datasource import DataSource
from adlframework.dataentity.midi_de import MidiDataEntity
### Model
from adlframework.nets.image_nets import mnist_test
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from adlframework.experiment import Experiment
### Controllers
from adlframework.processors.general_processors import reshape, make_categorical
### Callbacks
from adlframework.callbacks.image_callbacks import SaveValImages

import pdb


### Controllers
processors = [partial(reshape, out_shape=(28, 28, 1)),
			  partial(make_categorical, num_classes=10)]

### Load Data
midi_retrieval = GlobSearch('/Users/localhost/Desktop/Projects/Working/StudyMuse/local_cache', '*.[mM][xXiI][lLdD]')
midi_ds = DataSource(midi_retrieval, MidiDataEntity, processors=processors)

train_ds, temp = DataSource.split(midi_ds, split_percent=.6)
val_ds, test_ds = DataSource.split(temp, split_percent=.6)

### Load network
net = mnist_test(input_shape=(28, 28, 1), target_shape=10)

### Callbacks
callbacks = []

### Create and run experiment
exp = Experiment(train_datasource=train_ds,
					validation_datasource=val_ds,
					network=net,
					metrics=['mae', 'acc'],
					loss=categorical_crossentropy,
					optimizer=Adadelta(),
					label_names=list(range(10)),
					callbacks=callbacks,
					epochs=1)
exp.run()
