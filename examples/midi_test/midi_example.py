from functools import partial
### Data
from adlframework.retrievals.BlobLocalCache import BlobLocalCache
from adlframework.datasource import DataSource
from adlframework.dataentity.midi_de import MidiDataEntity
### Model
from midi_net import midi_test
from keras.optimizers import Adadelta
from keras.losses import MAE
from adlframework.experiment import Experiment
### Controllers
from adlframework.processors.general_processors import crop, reshape, pdb_trace
from adlframework.processors.lstm_processors import crop_and_label
from adlframework.processors.midi_processors import midi_to_np
from adlframework.filters.general_filters import min_array_shape
### Callbacks

### Controllers
controllers = [ #partial(threshold_label, labelnames="num_instruments", threshold=1, greater_than=False),
			  midi_to_np,
			  partial(min_array_shape, min_shape=(105, 4)),
			  partial(crop, shape=(105, 4)),
			  partial(crop_and_label, num_rows=5),
			  partial(reshape, reshape_label=True, out_shape=(20,)),
			  pdb_trace
			 ]

### Load Data
base = '/Users/localhost/Desktop/Projects/Working/StudyMuse/local_cache/alex_midiset/v2/'
midi_retrieval = BlobLocalCache(base+'midis/', base+'labels/')
midi_ds = DataSource(midi_retrieval, MidiDataEntity,
						controllers=controllers,
						batch_size=1,
						backend='madmom')

train_ds, temp = DataSource.split(midi_ds, split_percent=.6)
val_ds, test_ds = DataSource.split(temp, split_percent=.6)

### Load network
net = midi_test(input_shape=(100, 4), target_shape=20)

### Callbacks
callbacks = []

### Create and run experiment
exp = Experiment(train_datasource=train_ds,
					validation_datasource=val_ds,
					network=net,
					metrics=['mae'],
					loss=MAE,
					optimizer=Adadelta(),
					epochs=1)
exp.run()
