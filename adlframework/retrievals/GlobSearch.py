from adlframework.retrievals.retrieval import Retrieval
import os
from glob2 import iglob

class GlobSearch(Retrieval):
	'''
	Uses the glob module to find all files recursively matching description.
	Does not currently find labels
	'''
	_return_type = 'file path'
	def __init__(self, superdir, file_regex):
		super(GlobSearch, self).__init__()
		self.file_regex = file_regex
		self.superdir = os.path.abspath(superdir)+'/'

	def get_data(self, id_):
		'''
		Returns a file path.
		'''
		return id_

	def get_label(self, id_):
		'''
		Returns a file path.
		'''
		return None

	def list(self):
		'''
		Returns a list of unique string identifiers.
		'''
		return iglob(self.superdir+'**/'+self.file_regex)
