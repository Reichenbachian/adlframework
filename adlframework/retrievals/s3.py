from .retrieval import Retrieval
class S3Hive(Retrieval):
	'''
	This represents all data entities being in many directories.
	As separated by speakers.

	For instance:
	local_cache/
		validation/
			subclass_1/
				sample
				sample
			subclass_2/
				sample
			...
		training/
			subclass_1/
				sample
				sample
			subclass_2/
				sample
			...
	For comparison, check BlobLocalCache.
	'''
	def __init__(self, directory):
		super(S3Hive, self).__init__()
		raise NotImplemented('Not done!')

	def read(self, id_):
		raise NotImplemented('Not done!')

	def list(self, subclass=None):
		raise NotImplemented('Not done!')