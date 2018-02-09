'''
A dataentity represents a single piece of data.
For instance, it could be an audio file or an image.

No data loading should be done in the __init__ function.
Data Loading should be postponed as much as possible.

Required Functions
 - Either _read_raw, _read_np, or _read_file, otherwise it's rather useless.
 - get_sample, returns a sample

Required Properties
 - self.labels,    contains a pandas dataframe
 - self.data,      contains the data
 - self.retrieval, contains the retrieval object

Optional Functions
 - 

Optional Parameters
 - 

'''

class DataEntity(object):
	def __init__(self, retrieval, verbosity, backend='default'):
		self.verbosity = verbosity
		self.retrieval = retrieval
		self.backend = backend

	def _read_raw(self, id_):
		"""
		Should read the data directly from raw data.
		This is file dependent. It might be raw mp4 data
		or merely a file 
		"""
		raise NotImplemented('Reading from raw data is not implemented in %s.' % self.__class__)

	def _read_np(self, id_):
		"""
		Should read the data directly from a numpy array. 
		i.e: The retrieval will return a numpy array.
		"""
		raise NotImplemented('Reading from numpy data is not implemented in %s.' % self.__class__)

	def _read_file(self, id_):
		"""
		Should read the data directly from a file.
		"""
		raise NotImplemented('Reading from a file is not implemented in %s.' % self.__class__)

	def get_sample(self, id_):
		"""
		Returns a sample.
		THERE IS NO GUARANTEE THAT THIS SAMPLE OR LABEL IS THE SAME EVERY TIME.
		(Cannot guarantee due to potentially infinite samples.)
		Returns: np_arr, label
		"""
		raise NotImplemented('get_sample is not implemented in %s.' % self.__class__)

	def filter(self, func):
		'''
		Goes through each segment, as defined in self.labels 2nd dimension.
		Checks if segment matches qualification. If not, delete from label
		and request removal through remove_segment function, which isn't necessarily
		defined.

		Note: If sample does not use the self.labels and remove_segment is not defined
		a segment may still be sampled.

		To-Do: Give access to data in filter.
		'''
		for i, label in self.labels.iterrows():
			if not func(label):
				del self.labels.iloc[i]
				if hasattr(self, 'remove_segment'):
					self.remove_segment(i)

	def get_data(self, id_):
		"""
		Decides whether to read from a file or from raw based on retrieval and returns output.
		"""
		if self.retrieval.return_type().lower() == 'np array':
			return self._read_np(id_)
		elif self.retrieval.return_type().lower() == 'file path':
			return self._read_file(id_)
		elif self.retrieval.return_type().lower() == 'raw':
			return self._read_raw(id_)
		else:
			raise NotImplementedError("Retrieval's return type not supported.")
