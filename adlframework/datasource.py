from random import shuffle, choice
from multiprocessing import Pool, Queue
import numpy as np
import pandas as pd
import logging
import pdb
import copy

logger = logging.getLogger(__name__)

class DataSource():
	'''
	The point of a data source is to convert a retrieval into a bunch of accessible data entities.

	Constructor
	-----------
	 - 'retrieval' - the retrieval
	 - 'Entity' - the entity class for which to construct entities.

	Attributes
	-----------
	 - '_entities' - a list of data entities
	 - '_retrieval' - The retrieval for the data source
	'''

	def __init__(self, retrieval, Entity, filters=[], augmentors=[],
					processors=[], ignore_cache=False, batch_size=30, workers=1, **kwargs):
		self._retrieval = retrieval
		self.filters = filters
		self.augmentors = augmentors
		self.processors = processors
		self._entities = []
		self.queue = Queue()
		self.batch_size = batch_size
		self.list_pointer = 0
		self.multiprocessed = workers > 1
		if self.multiprocessed:
			self.pool = Pool(processes=workers)
		if not ignore_cache and retrieval.is_cached(): # Read from cache
			self._entities = self._retrieval.load_from_cache()
		else: # create cache otherwise
			for id_ in retrieval.list():
				self._entities.append(Entity(id_, retrieval, **kwargs))
			retrieval.cache()
		##### Prefilter!
		##### This is the point where all prefiltering based on entity occurs.
		##### To-Do: write this
		shuffle(self._entities)
		assert len(self._entities) > 0, "Cannot initialize an empty data source"
		assert type(self.augmentors) is list, "Please make augmentors a list in all data sources"
		assert type(self.filters) is list, "Please make filters a list in all data sources"
		assert type(self.processors) is list, "Please make processors a list in all data sources"

	def should_reset_queue(self):
		shuffle(self._entities)
		self.list_pointer = 0

	def process_sample(self, sample):
		'''
		Augments and processes a sample.
		A sample goes through a single augmentor(which may be a list of augmentors [aug1, aug2, ...])
			For instance, augmentors may be [aug1, aug2, [aug3, aug4]] and it may choose aug1, aug2, or [aug3, aug4].
		A sample goes through every processor.
		'''
		### Augment
		if len(self.augmentors) > 0:
			chosen_augmentor = choice(augmentors)
			chosen_augmentor_list = [augmentor] if type(augmentor) is not list else augmentor
			for augmentor in chosen_augmentor_list:
				sample = augmentor(sample)
		### Processor
		if len(self.processors) > 0:
			for processor in self.processors:
				sample = processor(sample)
		return sample

	def next(self, batch_size=None):
		'''
		Creates the next batch
		It will filter, process, and augment the data.
		If no controller has converted the labels to lists, then it will do so, assuming that
		all labels are used.

		Additionally, allows overwriting of batch_size constant.
		'''
		batch_size = batch_size if batch_size != None else self.batch_size
		should_reset_queue = False
		batch = []
		# Load batches
		for i in range(batch_size): # Create a batch
			entity = self._entities[self.list_pointer] # Grab next entity
			sample = entity.get_sample()
			if all([f(sample) for f in self.filters]): # Only add to batch if it passes all per sample filters
				batch.append(sample)
			self.list_pointer += 1
			if self.list_pointer >= len(self._entities): # Loop batch if necessary(while randomize before next iteration)
				self.list_pointer = 0
				should_reset_queue = True
		# Process batches
		if self.multiprocessed:
			self.pool.map(self.process_sample, batch)
			raise NotImplemented("multiprocessing not implemented")
		else:
			batch = map(self.process_sample, batch)
		# Reset entities if necessary
		if should_reset_queue:
			logger.log(logging.INFO, 'Looping the datasource')
			self.list_pointer = 0
			shuffle(self._entities)
		# Turn sample into keras readable sample
		data, labels = zip(*batch)
		labels = list(labels)
		if type(labels[0]) is pd.Series:
			for i in range(len(labels)):
				labels[i] = labels[i].tolist()
		data = np.array(data)
		labels = np.array(labels)
		return data, labels

	@staticmethod
	def split(ds1, split_percent=.5):
		'''
		Splits one datasource into two

		Returns: datasource_1, datasource_2
				 where len(datasource_1)/len(ds) approximately equals split_percent
		'''
		logger.log(logging.WARNING, 'Using split may cause datasource specific training. (For instance, overfitting on a single speaker.)')
		break_off = int(len(ds1._entities)*split_percent)
		shuffle(ds1._entities)
		### To-Do: Fix below inefficiency
		ds2 = copy.copy(ds1)
		ds2._entities = ds1._entities[break_off:]
		ds1._entities = ds1._entities[:break_off]
		return ds1, ds2

	def __next__(self, batch_size=None):
		'''
		Wrapper for python 3
		'''
		return self.next(batch_size)

