ADLFramework: An open-source deep learning framework

ADLFramework can currently take data from multiple retrieval types(Local File Systems, mysql databases, Json files), processes them through a series of controllers, and serves them in a simple python iterator.

It does this while easily multiprocessing, looking after memory constraints, and filtering the dataset.

## Easy Example
For instance, a simple example may be:
```
controllers = [partial(reshape, shape=(28, 28, 1)),
			  partial(make_categorical, num_classes=10)]
mnist_retrieval = MNIST_retrieval()
mnist_ds = DataSource(mnist_retrieval, ImageFileDataEntity, controllers=controllers)
batch = mnist_ds.next()
```
For more information on this example, check out examples/MNIST/mnist_example.py.

## Advanced Example
A more advanced example may look like:
```
prefilters = [partial(threshold_label, labelnames="num_instruments", threshold=1, greater_than=False)]
controllers = [midi_to_np,
			         partial(min_array_shape, min_shape=(105, 4)),
			         partial(crop, shape=(105, 3)),
			         partial(crop_and_label, num_rows=5),
			         make_time_relative, # Makes the time column relative to previous time
			         notes_to_classification
			        ]

### Load Data
base = '/Users/localhost/Desktop/Projects/Working/StudyMuse/local_cache/alex_midiset/v2/'
midi_retrieval = BlobLocalCache(base+'midis/', base+'labels/')
midi_ds = DataSource(midi_retrieval, MidiDataEntity,
						controllers=controllers,
						backend='madmom',
						batch_size=50,
            workers=10,
            max_mem_percent=.8,
            prefilters=prefilters)
train_ds, temp = DataSource.split(midi_ds, split_percent=.6) # Train at .6
val_ds, test_ds = DataSource.split(temp, split_percent=.6) # Val at .24, test at .16
```
For more information on this example, check out examples/midi_test/midi_example.py.

## Controllers

#### Pre-Filter
A pre-filter goes through every segment of every dataentity and removes those labels from the object that do not match the criteria. Through the `remove_segment` method, that segment is requested to be removed from the dataentity(this can help preserve memory). If all segments are removed, then the entity is removed. A filter is given the label as its only required argument, though it is allowed to require more. For instance,
```
def a_pre_filter(entity, n=10, k=2):
	...
	return True/False
```

A controller is something that controls the data stream leaving the iterator. It follows the syntax below and usually falls into one of three categories.
```
def a_controller(sample, n=10, k=2):
	...
	return True/False (if filtering) or sample (if sample should continue)
```

### Controllers types
There are several types of controllers, including those listed below.

#### Filters
A filter gets a sample as an argument and returns True/False as to whether a sample meets criteria. This is considerably slower than a prefilter.

#### Augmentors
Augmentors somehow change the data. Augmentors are usually only applied to the training data. 

#### Processors
Processors are usually applied to both the training and validation data. For instance, normalization is usually done here. If this datasource is being used as an input to an experiment object, then, currently, it must return a valid numpy array input to keras.

## Mission Purpose
Modularize deep learning to allow large scale experimentation organization relatively easily.

## Install
```
pip install git+ssh://git@github.com/Reichenbachian/adlframework.git
```
