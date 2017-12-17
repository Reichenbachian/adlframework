'''
Represents a singular experiment
'''
import attr
import logging
import pdb

logger = logging.getLogger(__name__)


@attr.s
class Experiment(object):
	train_datasource = attr.ib()
	network = attr.ib()
	epochs = attr.ib(default=100)
	optimizer = attr.ib(default=None)
	loss = attr.ib(default=None)
	metrics = attr.ib(default=[])
	num_workers = attr.ib(default=1)
	max_queue_size = attr.ib(default=10)
	validation_datasource = attr.ib(default=None)
	verbose = attr.ib(default=1)

	# Batches
	train_batch_size = attr.ib(default=100)
	val_batch_size = attr.ib(default=100)

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
		self.multiprocessed = self.num_workers > 1
		self.compile_network()

		epoch_num = 0
		while (epoch_num <= self.epochs):
			self.network.fit_generator(self.train_datasource,
									   self.train_batch_size,
									   epochs=self.epochs,
									   use_multiprocessing=self.multiprocessed,
									   workers=self.num_workers,
									   max_queue_size=self.max_queue_size,
									   validation_steps=self.val_batch_size,
									   initial_epoch=epoch_num
									   )
			epoch_num += 1
