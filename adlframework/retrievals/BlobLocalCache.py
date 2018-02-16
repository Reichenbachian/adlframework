from adlframework.retrievals.retrieval import Retrieval
import os
import pandas as pd

class BlobLocalCache(Retrieval):
	'''
	This represents all data entities being in one directory.
	For instance:
	local_cache/
		wav/
			validation/
				1.wav
				2.wav
				...
			training/
				4.wav
				100.wav
				...
		labels/
			validation/
				1.csv
				2.csv
				...
			training/
				4.csv
				100.csv
				...
	For comparison, check LocalHiveCache.
	'''
	_return_type = 'file path'
	def __init__(self, data_dir, labels_dir):
		super(BlobLocalCache, self).__init__()
		self.data_dir = os.path.abspath(data_dir)+'/'
		self.label_dir = os.path.abspath(labels_dir)+'/'

	def get_data(self, id_):
		'''
		Returns a file path.
		'''
		return self.data_dir+id_

	def get_label(self, id_):
		'''
		Returns a file path.
		'''
		return pd.read_csv(self.label_dir+id_[:id_.rfind('.')]+'.csv')

	def list(self):
		'''
		Returns a list of unique string identifiers.
		'''
		return os.listdir(self.data_dir)
