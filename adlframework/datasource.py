import pdb
from random import shuffle
from multiprocessing import Pool, Queue
import numpy as np
import logging
import copy
from adlframework.utils import in_ipynb
from datasource_union import DataSourceUnion
import psutil

# Import corrent version of tqdm(jupyter notebook vs non)
if in_ipynb():
	from tqdm import tqdm_notebook as tqdm
else:
	from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataSource():
	'''
	The point of a data source is to convert a retrieval into a bunch of accessible data entities.

	Constructor
	-----------
	 - 'retrieval' - the retrieval
	 - 'Entity' - the entity class for which to construct entities.
	 - 'timeout' - timeout is used on a per sample basis. Not per batch.

	Attributes
	-----------
	 - '_entities' - a list of data entities
	 - '_retrieval' - The retrieval for the data source
	'''

	def __init__(self, retrieval, Entity, controllers=[], ignore_cache=False, batch_size=30, timeout=None,
					prefilters=[], verbosity=logging.DEBUG, max_mem_percent=.95, **kwargs):
		logging.basicConfig(level=verbosity)
		self.max_mem_percent = max_mem_percent
		self.verbosity = verbosity	# 0: little to no debug, 1 some debug, 3 all debug.
		self._retrieval = retrieval
		self.controllers = controllers
		self.prefilters = prefilters
		self._entities = []
		self.batch_size = batch_size
		self.list_pointer = 0
		self.timeout = timeout
		if retrieval == None:
			logger.log(logging.INFO, 'retrieval is set to none. Assuming a single entity with random initialization.')
			self._entities.append(Entity(0, **kwargs))
		else:
			if not ignore_cache and retrieval.is_cached(): # Read from cache
				self._entities = self._retrieval.load_from_cache()
			else: # create cache otherwise
				for id_ in retrieval.list():
					self._entities.append(Entity(id_, retrieval, verbosity, **kwargs))
				retrieval.cache()

		self.__prefilter()

		shuffle(self._entities)
		assert len(self._entities) > 0, "Cannot initialize an empty data source"
		assert type(self.controllers) is list, "Please make augmentors a list in all data sources"
		assert type(self.prefilters) is list, "Please make augmentors a list in all data sources"

	def __prefilter(self):
		'''
		Prefilters the samples.
		A prefilter gets just the label and the id.
		### To-Do: Implement Remove_segment.
		'''
		### Prefilter
		NUM_DASHES = 40
		logger.log(logging.INFO, 'Prefiltering entities')
		for i, pf in enumerate(self.prefilters):
			logger.log(logging.INFO, '-'*NUM_DASHES+'Filter ' +str(i)+'-'*NUM_DASHES)
			logger.log(logging.INFO, pf)
			logger.log(logging.INFO, 'Before filter '+str(i)+', there are '+ str(len(self._entities))+' entities.')
			self._entities = filter(pf, tqdm(self._entities))
			logger.log(logging.INFO, 'After filter '+str(i)+', there are '+ str(len(self._entities))+' entities.')

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
		### Processor
		for controller in self.controllers:
			tmp = controller(sample)
			sample = sample if tmp == True else tmp
			if sample == False:
				return sample
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
		# pbar = tqdm(total=batch_size, leave=False)
		while len(batch) < batch_size: # Create a batch
			entity = self._entities[self.list_pointer] # Grab next entity

			try:
				sample = entity.get_sample()
				sample = self.process_sample(sample)
				if sample: # Only add to batch if it passes all per sample filters
					# To-Do: Somehow prevent redundant rejections.
					batch.append(sample)
					# pbar.update()
			except Exception as e:
				if self.verbosity == 3:
					logging.error('Controller or sample Failure')
					logging.error(e, exc_info=True)
			self.list_pointer += 1
			if self.list_pointer >= len(self._entities): # Loop batch if necessary(while randomize before next iteration)
				self.list_pointer = 0
				logger.log(logging.INFO, 'Looped the datasource')
				should_reset_queue = True
		
			# Check if we have enough memory to keep sample in memory
			mem = psutil.virtual_memory()
			if mem.percent/100.0 > self.max_mem_percent:
				del entity.data
			del mem # Shouldn't be necessary, but just in case.

		# Reset entities if necessary
		if should_reset_queue:
			logger.log(logging.INFO, 'Shuffling the datasource')
			self.list_pointer = 0
			shuffle(self._entities)

		# Turn sample into keras readable sample
		data, labels = zip(*batch)
		labels = list(labels)
		data = np.array(data)
		labels = np.array(labels)
		return data, labels

	def filter_ids(self, id_list):
		'''
		Filters by a list of ids
		'''
		id_set = set(id_list)
		def in_set(e):
			return e.unique_id in id_set
		logger.log(logging.INFO, 'Filtering by id list. Size pre-filter is'+str(len(self._entities)))
		self._entities = filter(in_set, self._entities)
		logger.log(logging.INFO, 'Done filtering. Size post-filter is'+str(len(self._entities)))

	def save_ids(self, name):
		'''
		Saves a list of newline delimiated ids.
		'''
		open(name, 'w').write('\n'.join([str(x.unique_id) for x in self._entities]))

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

	def __iter__(self):
		'''
		Making an iterator object
		'''
		return self

	def __next__(self, batch_size=None):
		'''
		Wrapper for python 3
		'''
		return self.next(batch_size)

	def __add__(self, other_dsa):
		"""
		Combines two datasource objects while maintaining percentages.
		"""
		if isinstance(other_dsa, DataSource):
			return DataSourceUnion([self, other_dsa])
		elif isinstance(other_dsa, DataSourceUnion):
			dss = other_dsa.datasources
			dss.extend(self)
			return DataSourceUnion(dss)
		else:
			raise Exception("Can only combine DataSource or DataSourceUnion objects!")
