## Data Source/Entity Pairing

The data source collection is an iterator for data entities.
Each data entity represents a piece of data.
This data may contain many segments.
A segment, plus its corresponding label, is returned as a tuple and called a sample.

### Data Entity
A data entity requires the two following properties

**self.data**: Any piece of data. For now, all experiments use a keras backend, and therefore the data must be capable of being converted into a numpy array for input.

**self.labels**: A label must be a pandas dataframe. It should be loaded in the `__init__` method. The second(downwards) dimension is a unique identifier that represents the segments. This label may be changed by the filters.

### Illustrate: If a static `illustrate` function exists for a data entity, then both the data entity and its corresponding data source can be told to `de.illustrate()`. This will call the entities function to illustrate. It should open a window that somehow describes that data entity.

### Sample to File: If a static `sample_to_file` function exists for a data entity, then both the data entity and its corresponding data source can be told to `de.sample_to_file()`. This will call the entities function `sample_to_file`. It should save a file that visually illustrates the data file.

To-Do: Give option of which controllers to execute(ie. stop at index i.)


## Controllers


### Pre-Filter (Not yet Implemented)
A pre-filter goes through every segment of every dataentity and removes those labels from the object that do not match the criteria. Through the `remove_segment` method, that segment is requested to be removed from the dataentity(this can help preserve memory). If all segments are removed, then the entity is removed. A filter is given the label as its only required argument, though it is allowed to require more. For instance,
```
def a_pre_filter(label, n=10, k=2):
	...
	return True/False
```

A controller is something that controls the data stream leaving the iterator. It follows the syntax below and usually falls into one of three categories.
```
def a_controller(sample, n=10, k=2):
	...
	return True/False (if filtering) or sample (if sample should continue)
```

### Filters
A filter gets a sample as an argument and returns True/False as to whether a sample meets criteria. This is considerably slower than a prefilter.

### Augmentors
Augmentors somehow change the data. Augmentors are usually only applied to the training data. 

### Processors
Processors are usually applied to both the training and validation data. For instance, normalization is usually done here. If this datasource is being used as an input to an experiment object, then, currently, it must return a valid numpy array input to keras.


## Batch Updates
A batch update is used to somehow edit the training or batch environment in between batches. A batch update is called in between batches. For instance, it may used as batch balancing or in gan training.

```
def a_batch_update(network, **kwargs):
	...
	return None # Should not return anything
```

Inside **kwargs are the network, batch... To-Do: Finish implementing batch-updates.


## Experiment Object
There are two types of experiment. Each represents an instance of an experiment and has the following methods.

 - `run()`: used to start experiment
 - `compile_network()`: used to compile the network if not already compiled.

 ### Simple Experiment
 To-Do: Write documentation

### Advanced Experiment
An advanced recordinig takes in an epoch method. It will run it for the specified number of epochs. Keras callbacks are still acceptable, though so are regular functions.