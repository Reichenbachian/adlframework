## Data Source/Entity Pairing

The data source collection is an iterator for data entities.
Each data entity represents a piece of data.
This data may contain many segments.
A segment, plus its corresponding label, is returned as a tuple and called a sample.

### Data Entity
A data entity requires the two following properties

**self.data**: Any piece of data. For now, all experiments use a keras backend, and therefore the data must be capable of being converted into a numpy array for input.

**self.labels**: A label must be a pandas dataframe. It should be loaded in the `__init__` method. The second(downwards) dimension is a unique identifier that represents the segments. This label may be changed by the filters.

### Pre-Filter
A pre-filter goes through every segment of every dataentity and removes those labels from the object that do not match the criteria. Through the `remove_segment` method, that segment is requested to be removed from the dataentity(this can help preserve memory). If all segments are removed, then the entity is removed. A filter is given the label as its only required argument, though it is allowed to require more. For instance,
```
def a_filter(label, n=10, k=2):
	...
	return True/False
```

### Filters
A filter gets a sample as an argument and returns True/False as to whether a sample meets criteria. This is considerably slower than a prefilter.
```
def a_filter(sample, n=10, k=2):
	...
	return True/False
```

### Augmentors
Augmentors somehow change the data. Augmentors are usually only applied to the training data. 
```
def an_augmentor(sample, arg1, n=10):
	...
	return segment, label
```

### Processors
Processors are usually applied to both the training and validation data. For instance, normalization is usually done here. If this datasource is being used as an input to an experiment object, then, currently, it must return a valid numpy array input to keras.

```
def a_processor(sample, arg1, arg2, n=10):
	...
	return valid_numpy_array, label
```