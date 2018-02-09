'''
Basic cache architecture
'''

class Cache(object):
	def __init__(self):
		raise NotImplemented("Don't directly initalize a Cache object, use a class in caches instead.")

	def has(self, id):
		raise NotImplemented('NOT IMPLEMENTED IN SUBCLASS!')

	def cache(self, id, data, label):
		raise NotImplemented('NOT IMPLEMENTED IN SUBCLASS!')

	def retrieve(self, id, label):
		raise NotImplemented('NOT IMPLEMENTED!')