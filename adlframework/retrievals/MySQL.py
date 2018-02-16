from adlframework.retrievals.retrieval import Retrieval
import os

class MySQL(Retrieval):
	'''
	This represents the data and their labels coming from a MySQL database.
	'''
	def __init__(self, data_dir, labels_dir):
		super(MySQL, self).__init__()
		self._return_type = 'file path'
		self.data_dir = os.path.abspath(data_dir)+'/'
		self.label_dir = os.path.abspath(labels_dir)+'/'

	def get_data(self, id_):
		'''
		Returns a file path.
		'''
		raise NotImplemented()

	def get_label(self, id_):
		'''
		Returns a file path.
		'''
		raise NotImplemented()

	def list(self):
		'''
		Returns a list of unique string identifiers.
		'''
		raise NotImplemented()
