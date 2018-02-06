'''
Represents the conglomeration of multiple data sources
union = ds1 + ds2
union = ds1 + different_union
'''
import math
import pdb
import numpy as np
from adlframework.utils import get_logger
import datasource
import copy

logger = get_logger()

class DataSourceUnion():

	def __init__(self, datasources):
		'''
		Takes an input of a list of datasource objects
		'''
		self.datasources = datasources
		self.batch_sizes = [x.batch_size for x in self.datasources]
		self.batch_size = sum(self.batch_sizes) # batch size of union is the sum of all its subsidiaries

	def split(self, split_percent=.5):
		'''
		Splits one DataSourceUnions into two

		To-Do: Write better comment and make more efficient
		'''
		dss1 = copy.copy(self)
		dss2 = copy.copy(self)
		dss1.datasources = []
		dss2.datasources = []
		for ds in self.datasources:
			ds1, ds2 = ds.split(split_percent)
			dss1.datasources.append(ds1)
			dss2.datasources.append(ds2)
		return dss1, dss2

	def next(self, percentages=None):
		'''
		Returns a batch taken from the datasources of the union.

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
			return np.array(batch_X), np.array(batch_y)

	def __add__(self, other_dsa):
		"""
		Combines either datasource_union +  datasource_union or
		datasource_union + datasource.
		"""
		if isinstance(other_dsa, datasource.DataSource):
			return DataSourceUnion(self.datasources[:]+[other_dsa])
		elif isinstance(other_dsa, DataSourceUnion):
			dss = other_dsa.datasources[:]  # Copy it
			return DataSourceUnion(dss+self.datasources)
		else:
			raise Exception("Can only combine DataSource or DataSourceUnion objects!")

