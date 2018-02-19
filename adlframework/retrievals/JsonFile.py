from adlframework.retrievals.retrieval import Retrieval
import pandas as pd
import numpy as np

class JsonFile(Retrieval):
	'''
	This represents all data entities being in one json file.
	For instance:
	local_cache/
		train.json
		validation.json

	json in this format
	Col1, Col2, Col3, Label1, Col4, Label2
	Returns a numpy array
	'''
	_return_type = "np array"
	def __init__(self, fp, data_columns, label_columns):
		super(JsonFile, self).__init__()
		self.df = pd.read_json(fp).reset_index()
		self.data_columns = data_columns
		self.label_columns = label_columns
		assert type(self.label_columns) is list, 'label_columns must be a list'
		assert type(self.data_columns) is list, 'data_columns must be a list'

	def get_data(self, id_):
		'''
		Returns data of image.

		Must convert twice because 'as_matrix' does not return a matrix, as per numpy documentation.
		TO-DO: Find a better conversion method.
		'''
		return np.array(self.df.as_matrix(self.data_columns)[id_].tolist())

	def get_label(self, id_):
		'''
		Returns a pandas dataframe
		'''
		return self.df.loc[[id_]][self.label_columns]

	def list(self):
		'''
		Returns a list of unique string identifiers.
		'''
		return self.df.index
