'''
Represents a singular experiment
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import attr
from keras.callbacks import TensorBoard, LambdaCallback
import hashlib
from random import random
import pdb
from tqdm import tqdm
from adlframework.utils import get_logger

logger = get_logger()

class __Experiment__(object):
	def compile_network(self):
		if self.loss == None:
			logger.warn('No loss is defined. Will default to mse. Be warned that leaving this default can be a bad idea!!!')
			self.loss = 'mse'
		if self.optimizer == None:
			logger.warn('No optimizer is defined. Will default to rmsprop.')
			self.optimizer = 'rmsprop'
		self.network = self.network.build_model()
		assert self.network != None, 'Make sure to return model in build_model construction'
		self.network.compile(optimizer=self.optimizer,
							 loss=self.loss,
							 metrics=self.metrics)
		logger.info('Compiled network.')

@attr.s
class SimpleExperiment(__Experiment__):
	train_datasource = attr.ib()
	network = attr.ib()
	
	epochs = attr.ib(default=100)
	optimizer = attr.ib(default=None)
	callbacks = attr.ib(default=[])
	metrics = attr.ib(default=[])
	loss = attr.ib(default=None)
	label_names = attr.ib(default=None)
	max_queue_size = attr.ib(default=10)
	workers = attr.ib(default=1)
	verbose = attr.ib(default=1)
	disable_tensorboard = attr.ib(default=False)
	use_multiprocessing = attr.ib(default=False)
	validation_datasource = attr.ib(default=None)
	test_datasource = attr.ib(default=None)
	tb_dir = attr.ib(default='./exps/exp_'+hashlib.md5(str.encode(str(random()))).hexdigest())


	# Batches
	train_batch_steps = attr.ib(default=30)
	val_batch_steps = attr.ib(default=30)

	def run(self):
		#### Create output folder
		if not os.path.exists(self.tb_dir):
			os.makedirs(self.tb_dir)

		### Give all callbacks access to the experiment object
		for callback in self.callbacks:
			callback.exp = self

		### Compile the network
		self.compile_network()
		
		### Enable tensorflow callback by default
		if not self.disable_tensorboard:
			self.callbacks.append(TensorBoard(log_dir=self.tb_dir, histogram_freq=0,  
										write_graph=True, write_images=True))

		### Create mutliprocessing constant if not already existent
		if self.use_multiprocessing == None:
			self.use_multiprocessing = self.workers > 1

		### Train
		self.network.fit_generator(self.train_datasource,
									   self.train_batch_steps,
									   epochs=self.epochs,
									   use_multiprocessing=self.use_multiprocessing,
									   workers=self.workers,
									   validation_data=self.validation_datasource,
									   validation_steps=self.val_batch_steps,
									   callbacks=self.callbacks
								   )

		### Test network on test batch
		if self.train_datasource is not None:
			test_out = self.network.test_on_batch(*self.train_datasource.next())
			metrics = self.network.metrics
			metrics.insert(0, 'loss')
			print("Test metrics: ", zip(metrics, test_out))

@attr.s
class AdvancedExperiment(__Experiment__):
	epoch = attr.ib()
	network = attr.ib()

	epochs = attr.ib(default=100)
	loss = attr.ib(default=None)
	metrics = attr.ib(default=None)
	optimizer = attr.ib(default=None)
	callbacks = attr.ib(default=[])
	should_compile_network = attr.ib(default=False)
	tb_dir = attr.ib(default='./exps/exp_'+hashlib.md5(str.encode(str(random()))).hexdigest())

	def run(self):
		#### Create output folder
		if not os.path.exists(self.tb_dir):
			os.makedirs(self.tb_dir)

		### Give all callbacks access to the experiment object
		for callback in self.callbacks:
			callback.exp = self

		### Compile the network
		self.compile_network()
		
		for i in range(len(self.callbacks)):
			callback = self.callbacks[i]
			if type(callback) is type(self.run): # Is it a function
				self.callbacks[i] = LambdaCallback()


		for epoch in tqdm(range(self.epochs)):
			##### On Epoch begin
			info_dict = self.epoch()

