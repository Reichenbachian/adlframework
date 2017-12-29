from functools import partial
### Data
from adlframework.retrievals.BlobLocalCache import BlobLocalCache
from adlframework.datasource import DataSource
from adlframework.dataentity.midi_de import MidiDataEntity
import numpy as np
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
from keras.callbacks import ModelCheckpoint

def make_time_relative(sample):
	data, label = sample
	tmp = data
	for tmp in [data, label]:
		for i in range(len(tmp)-1, 0, -1):
			tmp[i][0] = tmp[i][0] - tmp[i-1][0]# Assume timestamps are at 0
		tmp[0][0] = 0
		tmp[:,1] = tmp[:,1]/88.0
	return data, label.reshape((15,))

### Controllers
controllers = [ #partial(threshold_label, labelnames="num_instruments", threshold=1, greater_than=False),
			  pdb_trace,
			  midi_to_np,
			  partial(min_array_shape, min_shape=(105, 4)),
			  partial(crop, shape=(105, 3)),
			  partial(crop_and_label, num_rows=5),
			  (lambda s: (s[0][:,:3], s[1])), ## Selects only [Onset time, pitch, duration] and normalize
			  make_time_relative, # Makes the time column relative to previous time
			  (lambda s: ((np.apply_along_axis(abs, 1, s[0]) < 10).all() and (np.apply_along_axis(abs, 0, s[1]) < 10).all())) # Make sure all are reasonable
			 ]

### Load Data
base = '/Users/localhost/Desktop/Projects/Working/StudyMuse/local_cache/alex_midiset/v2/'
midi_retrieval = BlobLocalCache(base+'midis/', base+'labels/')
midi_ds = DataSource(midi_retrieval, MidiDataEntity,
						controllers=controllers,
						backend='madmom')

train_ds, temp = DataSource.split(midi_ds, split_percent=.6) # Train at .6
val_ds, test_ds = DataSource.split(temp, split_percent=.6) # Val at .24, test at .16

### Load network
net = midi_test(input_shape=(100, 3), target_shape=15)

### Callbacks
callbacks = [ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')]

### Create and run experiment
exp = Experiment(train_datasource=train_ds,
					validation_datasource=val_ds,
					test_datasource=test_ds,
					callbacks=callbacks,
					network=net,
					metrics=['mae'],
					loss=MAE,
					optimizer=Adadelta(),
					epochs=1000)
exp.run()
