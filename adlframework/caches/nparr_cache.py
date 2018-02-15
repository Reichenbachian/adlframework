import numpy as np
import pdb
import pickle
import os
from adlframework.cache import Cache
from adlframework.utils import get_logger

logger = get_logger()

class RegularNPArrCache(Cache):
	'''
	TO-DO: Written for 1-d. Generalize to N-D.
	'''
	def __init__(self, cache_file=None, compress=True):
		self.data = None
		self.labels = None
		self.id_to_index = {}
		self.data_size = -1
		self.label_size = -1
		self.c_index = 1 # Location of end of array
		self.cache_file = cache_file
		self.compress = compress
		if cache_file is not None:
			self.load()

	''' Necessary classes '''
	def has(self, id_):
		'''
		Checks if cached. In this format to maintain O(1) lookup.
		'''
		try:
			self.id_to_index[id_]
			return True
		except:
			return False

	def cache(self, id_, data, label):
		# Warning! Inefficient. Copying memory over each time.
		# To-Do: Initialize block of memory and don't go over or under
		### Check sizes
		if self.data_size == -1: # Data
			self.data_size = len(data)
		else:
			assert(self.data_size) == len(data), 'Variable Lengths data segments were received to RegularNPArrCache cache!'
		if self.label_size == -1: # Label
			self.label_size = len(label)
		else:
			assert(self.label_size) == len(label), 'Variable Lengths labels were received to RegularNPArrCache cache!'
		### Read into cache
		if type(self.data) == type(None):
			self.data = np.array([data])
			self.labels = np.array([label])
		else:
			assert len(self.data) == len(self.labels), "Internal error in RegularNPArrCache"
			try:
				if self.c_index >= len(self.data):
					self.double_arr_size()
				self.data[self.c_index] = data
				self.labels[self.c_index] = label
				self.c_index += 1
			except Exception as e:
				logger.error(str(e))
		self.id_to_index[id_] = len(self.data) - 1
		return True

	def retrieve(self, id_):
		idx = self.id_to_index[id_]
		return self.data[idx], self.label[idx]


	def save(self):
		'''
		Saves Object
		'''
		if self.cache_file != None:
			self.crop_excess()
			if self.compress:
				np.savez_compressed(self.cache_file+'_joined', self.data, self.labels)
			else:
				np.save(self.cache_file+'_data', self.data, allow_pickle=True, fix_imports=True)
				np.save(self.cache_file+'_label', self.labels, allow_pickle=True, fix_imports=True)
			pickle.dump( self.id_to_index, open(self.cache_file+'_dict', "wb" ))

	def load(self):
		'''
		Saves data numpy array to a file.
		Currently, only saves data.
		To-Do: save labels too.
		'''
		dtf = self.cache_file+'_data'
		lf = self.cache_file+'_label'
		df = self.cache_file+'_dict'
		if self.cache_file != None and os.path.exists(dtf) and os.path.exists(lf) and os.path.exists(df):
			self.data = np.load(dtf)
			self.labels = np.load(lf)
			pickle.loads(open( "save.p", "wb" ))


	''' Extra Classes '''
	def crop_excess(self):
		self.data = self.data[:self.c_index]
		self.labels = self.labels[:self.c_index]

	def double_arr_size(self):
		new_d_shape = tuple([val if i != 0 else val*2 for i, val in enumerate(self.data.shape)])
		new_l_shape = tuple([val if i != 0 else val*2 for i, val in enumerate(self.labels.shape)])
		self.new_data = np.zeros(new_d_shape)
		self.new_labels = np.zeros(new_l_shape)
		self.new_data[:len(self.data)] = self.data
		self.new_labels[:len(self.labels)] = self.labels
		self.data = self.new_data
		self.labels = self.new_labels

class IrregularNPArrCache(Cache):
	'''
	TO-DO: Written for 1-d. Generalize to N-D.
	Reference: https://kastnerkyle.github.io/posts/using-pytables-for-larger-than-ram-data-processing/
	'''
	def __init__(self, cache_file=None, compress=True):
		self.data = []
		self.labels = []
		self.id_to_index = {}
		self.cache_file = cache_file


	''' Necessary classes '''
	def has(self, id_):
		'''
		Checks if cached. In this format to maintain O(1) lookup.
		'''
		try:
			self.id_to_index[id_]
			return True
		except:
			return False

	def cache(self, id_, data, label):
		### Read into cache
		self.data.append(data)
		self.labels.append(label)
		self.id_to_index[id_] = len(self.data) - 1

	def retrieve(self, id_):
		idx = self.id_to_index[id_]
		return self.data[idx], self.labels[idx]

	def load(self):
		'''
		Reads data, labels, and id_to_index as tuple from pickle
		'''
		if self.cache_file != None:
			if os.path.exists(self.cache_file):
				with open(self.cache_file, "wb") as f:
					self.data, self.labels, self.id_to_index = pickle.load(f)
			else:
				logger.warn('Cache file specified doesn\'t exist. Will continue...')

	def save(self):
		'''
		Save data, labels, and id_to_index as tuple in pickle
		'''
		if self.cache_file != None:
			with open(self.cache_file, "wb") as f:
				pickle.dump((self.data, self.labels, self.id_to_index), f)
		else:
			logger.warn('No cache file specified. Will lose cache on exit.')
