from adlframework.retrievals.retrieval import Retrieval
import glob
import os

class LocalHiveCache(Retrieval):
	'''
	This represents all data entities being in many directories.
	As separated by speakers.

	For instance:
	superfolder/
		subclass_1/
			sample
			sample
		subclass_2/
			sample
			...
	For comparison, check BlobLocalCache.
	'''
	def __init__(self, directory):
		super(HiveLocalCache, self).__init__()
		self.directory = os.path.abspath(directory)+'/'

	def get_label(self, id_):
		'''
		Returns a file.
		'''
		return open(self.directory+id_)

	def get_data(self, id_):
		'''
		Returns a file.
		'''
		return open(self.directory+id_)

	def list(self, subclass=None):
		if subclass == None:
			return glob.glob('./*/*')
		else:
			return glob.glob('./'+str(subclass)+'/*')
