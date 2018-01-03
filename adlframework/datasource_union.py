'''
Represents the conglomeration of multiple data sources
union = ds1 + ds2
union = ds1 + different_union
'''
import logging
import math
import pdb

logger = logging.getLogger(__name__)

class DataSourceUnion():

	def __init__(self, datasources):
		'''
		Takes an input of a list of datasource objects
		'''
		self.datasources = datasources
		self.batch_sizes = [x.batch_size for x in self.datasources]
		self.batch_size = sum(self.batch_sizes) # batch size of union is the sum of all its subsidiaries

	def next(self, percentages=None):
		'''
		Returns a batch taken from the datasources of the conglomerate.

		If percentages is defined, each batch is composed of the percent of each
		datasource as is defined by percentages.

		Otherwise, it will use each datasource's defined batch_size.
		'''
		while True:
			batch_sizes = [] # The batch sizes for each datasource
			if percentages is not None:
				assert len(percentages) == len(self.datasources), "Length of balanced percentages must be the same as the length of the data sources."
				batch_sizes = [math.ceil(p*self.batch_size) for p in percentages]
			else:
				batch_sizes = self.batch_sizes
			batch_X = []
			batch_y = []
			for i in range(len(self.datasources)):
				bs = batch_sizes[i]
				ds = self.datasources[i]
				batch = ds.next(bs)
				batch_X.extend(batch[0])
				batch_y.extend(batch[1])
				pdb.set_trace()
			return batch_X, batch_y

