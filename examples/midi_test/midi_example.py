from functools import partial
### Data
from adlframework.retrievals.BlobLocalCache import BlobLocalCache
from adlframework.datasource import DataSource
from adlframework.dataentity.midi_de import MidiDataEntity
import numpy as np
### Model
from midi_net import midi_test
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from adlframework.experiment import SimpleExperiment
### Controllers
from adlframework.processors.general_processors import crop, reshape, pdb_trace
from adlframework.processors.lstm_processors import crop_and_label
from adlframework.processors.midi_processors import midi_to_np, notes_to_classification, make_time_relative
from adlframework.filters.general_filters import min_array_shape
### Callbacks
from keras.callbacks import ModelCheckpoint

### Controllers
controllers = [ #partial(threshold_label, labelnames="num_instruments", threshold=1, greater_than=False),
			  midi_to_np,
			  partial(min_array_shape, min_shape=(105, 4)),
			  partial(crop, shape=(105, 3)),
			  partial(crop_and_label, num_rows=5),
			  make_time_relative, # Makes the time column relative to previous time
			  notes_to_classification
			 ]

### Load Data
# BUG: alex_midiset is stored, I assume, on alex's machine. 
base = '/Users/localhost/Desktop/Projects/Working/StudyMuse/local_cache/alex_midiset/v2/'
midi_retrieval = BlobLocalCache(base+'midis/', base+'labels/')
midi_ds = DataSource(midi_retrieval, MidiDataEntity,
						controllers=controllers,
						backend='madmom',
						batch_size=50)

train_ds, temp = DataSource.split(midi_ds, split_percent=.6) # Train at .6
val_ds, test_ds = DataSource.split(temp, split_percent=.6) # Val at .24, test at .16

### Load network
## Changing to classification task. 88 notes x 11 note types(durations) x 12 relative onset times
'''
Note types: Whole, half, quarter, eighth, sixteenth, third, sixth, seventh, dotted Whole, dotted half, dotted quarter
Onset Times: Note types + 0 onset(chord)
'''

net = midi_test(input_shape=(100, 88, 12, 11), target_shape=(5, 88, 12, 11))

### Callbacks
callbacks = [ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')]

### Create and run experiment
exp = SimpleExperiment(train_datasource=train_ds,
					validation_datasource=val_ds,
					test_datasource=test_ds,
					callbacks=callbacks,
					network=net,
					metrics=['mae', 'acc'],
					loss=categorical_crossentropy,
					optimizer=Adadelta(),
					epochs=1000,
					workers=3,
					use_multiprocessing=True,
					train_batch_steps=10,
					val_batch_steps=5)
exp.run()
