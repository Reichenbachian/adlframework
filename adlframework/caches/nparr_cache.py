from adlframework.cache import Cache
import numpy as np
import pdb

class RegularNPArrCache(Cache):
	def __init__(self):
		self.data = None
		self.labels = None
		self.id_to_index = {}
		self.size = -1

	def has(self, id_):
		'''
		Checks if cached. In this format to maintain O(1) lookup.
		'''
		try:
			self.data[id_]
			return True
		except:
			return False

	def cache(self, id_, data, label):
		# Warning! Inefficient. Copying memory over each time.
		# To-Do: Initialize block of memory and don't go over or under
		if self.size == -1:
			self.size = len(data)
		else:
			assert(self.size) == len(data), 'Variable Lengths were received to RegularNPArrCache cache!'
		if type(self.data) is None:
			self.data = np.array([data])
			self.labels = np.array([labels])
		else:
			pdb.set_trace()
			self.data = np.vstack([self.data, data])
			self.labels = np.vstack([self.labels, label])
		self.id_to_index[id_] = len(self.data) - 1
		assert len(self.data) == len(self.labels), "Internal error in RegularNPArrCache"
		return True


	def retrieve(self, id_):
		idx = self.id_to_index[id_]
		return self.data[idx], self.label[idx]
