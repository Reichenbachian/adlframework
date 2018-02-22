import pdb
from random import shuffle
import numpy as np
import copy
from adlframework.utils import in_ipynb
from adlframework.datasource_union import DataSourceUnion
import psutil
from adlframework.utils import get_logger
from adlframework.cache import Cache
# Import corrent version of tqdm(jupyter notebook vs non)
if in_ipynb():
	from tqdm import tqdm_notebook as tqdm
else:
	from tqdm import tqdm

logger = get_logger()

class DataSource():
	'''
	The point of a data source is to convert a retrieval into a bunch of accessible data entities.

	Constructor
	-----------
	 - 'retrieval' - the retrieval
	 - 'Entity' - the entity class for which to construct entities.
	 - 'timeout' - timeout is used on a per sample basis. Not per batch.

	 Multiprocessing
	 -------------------
	 - 'workers': number of asyncronous workers to fetch and process data. Does not apply to
				  prefiltering.
	 - 'queue_size': Number of samples the workers should prefetch.

	Attributes
	-----------
	 - '_entity_ids' - a list of data entity ids
	 - '_retrieval' - The retrieval for the data source


	To-Do: Cache processed/augmented results?
	To-Do: Compress memory cache?
	'''

	def __init__(self, retrieval, DataEntity, controllers=[], ignore_retrieval_cache=False,
					batch_size=30, timeout=None, prefilters=[], verbosity=3, max_mem_percent=.95, workers=1, queue_size=None,
					convert_batch_to_np=True, preload_memory=False, **kwargs):
		#### PRE-INITIALIZATION CHECKS
		assert type(controllers) is list, "Please make augmentors a list in all data sources"
		assert type(prefilters) is list, "Please make augmentors a list in all data sources"
		assert workers > 0, "Workers must be a positive integer."
		assert not (workers == 1 and queue_size != None), 'Queue_Size is only applicable to multiple workers. Try limiting memory with max_mem_percent.'

		#### CLASS VARIABLE INITIALIZATION
		self.max_mem_percent = max_mem_percent
		self.verbosity = verbosity	# 0: little to no debug, 1 some debug, 3 all debug.
		self._retrieval = retrieval
		self.controllers = controllers
		self.prefilters = prefilters
		self._entity_ids = []
		self.batch_size = batch_size
		self.list_pointer = 0
		self.timeout = timeout
		self.workers = workers
		self.convert_batch_to_np = convert_batch_to_np
		self.DE = DataEntity(retrieval, verbosity, **kwargs)

		self.cache_location = self.initialize_cache_location()
		assert not (self.cache_location == -1 and preload_memory), "No cache detected. Cannot preload memory."
		self.initialize_retrieval(ignore_retrieval_cache)
		self.__prefilter()
		if preload_memory:
			process_wrap = lambda x: self.process_id(x, just_cache=True)
			if workers != 1:
                                from multiprocessing import Pool
				with Pool(workers) as p:
					tqdm(p.imap(process_wrap, self._entity_ids), total=len(self._entity_ids))
			else:
				tqdm(map(process_wrap, self._entity_ids), total=len(self._entity_ids))
			self.cache.save()

		if self.workers > 1:
			self.initialize_multiprocessing(queue_size)


		#### POST-INITIALIZATION CHECKS
		assert len(self._entity_ids) > 0, "Cannot initialize an empty data source"

	def initialize_cache_location(self):
		'''
		Finds cache in controllers, if present.
		'''
		cache_locations = [issubclass(x.__class__, Cache) for x in self.controllers]
		assert sum(cache_locations) <= 1, "There should only be one cache object in controllers."     
		try:
			cache_index = cache_locations.index(True)
		except ValueError:
			return -1
		self.cache = self.controllers[cache_index]
		return cache_index

	def initialize_multiprocessing(self, queue_size):
		from multiprocessing import Process, Queue
		self.entity_queue = Queue(queue_size) # Stores entites. Not list indices.
		self.sample_queue = Queue(queue_size)
		Process(target=self.async_fill_queue).start()
		for _ in range(self.workers):
			Process(target=self.async_add_to_sample_queue).start()

	def initialize_retrieval(self, ignore_retrieval_cache):
		#### RETRIEVAL INITIALIZATION
		if self._retrieval == None:
			logger.info('retrieval is set to none. Assuming a single entity with random initialization.')
			self._entity_ids.append(0)
		else:
			if not ignore_retrieval_cache and self._retrieval.is_cached(): # Read from cache
				self._entity_ids = self._retrieval.load_from_cache()
			else: # create cache otherwise
				for id_ in self._retrieval.list():
					self._entity_ids.append(id_)
				self._retrieval.cache()
		shuffle(self._entity_ids)

	def async_fill_queue(self):
		while True: # Worst case: Very fast processor, very large queue. Then slow first load.
			self.entity_queue.put(self._entity_ids[self.list_pointer])
			self.list_pointer += 1
			if self.list_pointer >= len(self._entity_ids):
				self.reset_queue()

	def async_add_to_sample_queue(self):
		'''
		Runs continuously in background to collect data segments
		and add them to the common queue.
		'''
		while True:
			try:
				id_ = self.entity_queue.get()
				tmp = self.process_id(id_)
				sample = sample if tmp == True else tmp
				if sample: # If sample is processed and acceptable, append to queue
					self.sample_queue.put(sample)
			except Exception as e:
				if self.verbosity == 3:
					logger.error('Controller or sample Failure')
					logger.error(e, exc_info=True)

	def __prefilter(self):
		'''
		Prefilters the samples.
		A prefilter gets just the label and the id.
		### To-Do: Implement Remove_segment.
		'''
		### Prefilter
		NUM_DASHES = 40
		if len(self.prefilters) > 0:
			logger.info('Prefiltering entities')
		for i, pf in enumerate(self.prefilters):
			logger.info('-'*NUM_DASHES+'Filter '+str(i)+'-'*NUM_DASHES)
			logger.info(pf)
			logger.info('Before filter '+str(i)+', there are '+ str(len(self._entity_ids))+' entities.')
			def filter_wrapper(item):
				return pf(self.DE, item)
			## Pass the tuple DE, id to a filter.
			self._entity_ids = filter(filter_wrapper, tqdm(self._entity_ids))
			logger.info('After filter '+str(i)+', there are '+ str(len(self._entity_ids))+' entities.')

	def reset_queue(self):
		logger.info('Shuffling the datasource')
		shuffle(self._entity_ids)
		self.list_pointer = 0

	def process_id(self, id_, just_cache=False):
		'''
		Augments and processes a sample.
		A sample goes through a single augmentor(which may be a list of augmentors [aug1, aug2, ...])
			For instance, augmentors may be [aug1, aug2, [aug3, aug4]] and it may choose aug1, aug2, or [aug3, aug4].
		A sample goes through every processor.
		'''
		c_cont = 0
		sample = None

		### Read from cache
		if self.cache_location != -1 and self.cache.has(id_):
			try: # Try getting from cache
				sample = self.cache.retrieve(id_)
				c_cont = self.cache_location+1
			except:
				sample = self.DE.get_sample(id_)
		else:
			sample = self.DE.get_sample(id_)

		### Processor
		while c_cont < len(self.controllers):
			# Cache if present
			if c_cont == self.cache_location:
				self.cache.cache(id_, sample[0], sample[1])
				if just_cache:
					return
			else:
				# Run process
				controller = self.controllers[c_cont]
				tmp = controller(sample)
				sample = sample if tmp == True else tmp
				if sample == False:
					return False
			c_cont += 1
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
		batch = []
		while len(batch) < batch_size: # Create a batch
			if self.workers == 1:
				try:
					id_ = self._entity_ids[self.list_pointer] # Grab next entity
				except:
					pdb.set_trace()
				try:
					sample = self.process_id(id_)
					if sample: # Only add to batch if it passes all per sample filters
						# To-Do: Somehow prevent redundant rejections.
						batch.append(sample)

				except Exception as e:
					if self.verbosity == 3:
						logger.error('Controller or sample Failure')
						logger.error(e, exc_info=True)
				self.list_pointer += 1

			else:
				# Multiprocessing: grab from worker queue
				sample = self.sample_queue.get()
				batch.append(sample)

			if self.list_pointer >= len(self._entity_ids):
				self.reset_queue()

		data, labels = zip(*batch)
		if self.convert_batch_to_np:
			labels = np.array(labels)
			data = np.array(data)
		return data, labels  # Equivalent to data, labels

	def split(self, split_percent=.5):
		'''
		Splits one datasource into two

		Returns: datasource_1, datasource_2
				 where len(datasource_1)/len(ds) approximately equals split_percent
		'''
		logger.warn('Using split may cause datasource specific training. (For instance, overfitting on a single speaker.)')
		break_off = int(len(self._entity_ids)*split_percent)
		### To-Do: Fix below inefficiency
		ds1 = copy.copy(self)
		ds1.list_pointer = 0
		shuffle(ds1._entity_ids)
		ds2 = copy.copy(self)
		ds2._entity_ids = self._entity_ids[break_off:]
		ds1._entity_ids = self._entity_ids[:break_off]
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
			dss = other_dsa.datasources[:] # Copy it
			dss.extend(self)
			return DataSourceUnion(dss)
		else:
			raise Exception("Can only combine DataSource or DataSourceUnion objects!")
