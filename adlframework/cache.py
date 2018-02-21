'''
Basic cache architecture
'''
from adlframework.utils import get_logger

logger = get_logger()

class Cache(object):
	def __init__(self):
		raise NotImplemented("Don't directly initalize a Cache object, use a class in caches instead.")

	def has(self, id_):
		'''
		Checks if the id has been cached.
		Returns: True/False
		'''
		raise NotImplemented('has is not implemented in %s.' % self.__class__)

	def delete(self, id_):
		'''
		Checks if the id has been cached.
		Returns: True/False
		'''
		raise NotImplemented('delete is not implemented in %s.' % self.__class__)

	def cache(self, id_, data, label):
		'''
		Caches an id_ with its corresponding data and label.
		'''
		raise NotImplemented('cache is not implemented in %s.' % self.__class__)

	def retrieve(self, id_):
		'''
		Retrieves an id_.
		Returns a sample(i.e. (data, label))
		'''
		raise NotImplemented('retrieve is not implemented in %s.' % self.__class__)

	def load(self):
		'''
		Loads from a file/location
		'''
		logger.warning('load is not implemented in %s.' % self.__class__)

	def save(self):
		'''
		Saves to a file/location
		'''
		logger.warning('save is not implemented in %s.' % self.__class__)
