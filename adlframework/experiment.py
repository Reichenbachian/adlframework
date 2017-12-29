'''
Represents a singular experiment
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import attr
import logging
from keras.callbacks import TensorBoard
import hashlib
from random import random
import pdb

logger = logging.getLogger(__name__)


@attr.s
class Experiment(object):
	train_datasource = attr.ib()
	network = attr.ib()
	
	label_names = attr.ib(default=None)
	epochs = attr.ib(default=100)
	optimizer = attr.ib(default=None)
	callbacks = attr.ib(default=[])
	loss = attr.ib(default=None)
	metrics = attr.ib(default=[])
	workers = attr.ib(default=1)
	max_queue_size = attr.ib(default=10)
	validation_datasource = attr.ib(default=None)
	test_datasource = attr.ib(default=None)
	verbose = attr.ib(default=1)
	disable_tensorboard = attr.ib(default=False)
	use_multiprocessing = attr.ib(default=False)
	tb_dir = attr.ib(default='./exps/exp_'+hashlib.md5(str.encode(str(random()))).hexdigest())


	# Batches
	train_batch_steps = attr.ib(default=30)
	val_batch_steps = attr.ib(default=30)

	def compile_network(self):
		if self.loss == None:
			logger.log(
				logging.WARN, 'No loss is defined. Will default to mse. Be warned that leaving this default can be a bad idea!!!')
			self.loss = 'mse'
		if self.optimizer == None:
			logger.log(
				logging.WARN, 'No optimizer is defined. Will default to rmsprop.')
			self.optimizer = 'rmsprop'
		self.network = self.network.build_model()
		self.network.compile(optimizer=self.optimizer,
							 loss=self.loss,
							 metrics=self.metrics)
		logger.log(logging.INFO, 'Compiled network.')

	def run(self):
		if not os.path.exists(self.tb_dir):
			os.makedirs(self.tb_dir)
		for callback in self.callbacks: ### Give all callbacks access to the experiment object
			callback.exp = self

		self.compile_network()
		if not self.disable_tensorboard:
			self.callbacks.append(TensorBoard(log_dir=self.tb_dir, histogram_freq=0,  
										write_graph=True, write_images=True))
		if self.use_multiprocessing == None:
			self.use_multiprocessing = self.workers > 1
		### Run a minibatch
		self.network.fit_generator(self.train_datasource,
									   self.train_batch_steps,
									   epochs=self.epochs,
									   use_multiprocessing=self.use_multiprocessing,
									   workers=self.workers,
									   validation_data=self.validation_datasource,
									   validation_steps=self.val_batch_steps,
									   callbacks=self.callbacks
								   )
		test_out = self.network.test_on_batch(*self.train_datasource.next())
		metrics = self.network.metrics
		metrics.insert(0, 'loss')
		print("Test metrics: ", zip(metrics, test_out))
		# if self.test_datasource is not None:
		# 	self.network.
