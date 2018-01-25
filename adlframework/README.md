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


## Batch Updates (Not Implemented)
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
