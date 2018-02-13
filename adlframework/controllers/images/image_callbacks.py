from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import keras.backend as K

def fig2data(fig):
	'''
	Converts a figure to data.
	Credits to http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image.
	'''
	fig.canvas.draw()

	w, h = fig.canvas.get_width_height()
	buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	buf.shape = (h, w, 3)
	return buf

def convert_array_to_figure(data, label):
	'''
	Converts an array to a figure
	'''
	figure = plt.figure()
	plt.plot(data)
	plt.title(label)
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	return figure

class SaveValImages(Callback):
	"""
	Saves the validation images for each class every time is called.
	Parameters:
	----------
	log_dir:    Tensorboard directory to save the event files
	label_names: Names of labels in the order of network outputs
	interval : How often a light curve will be saved.
	"""

	def __init__(self, interval = 1):
		self.writer      = None
		self.interval    = interval
		
	def set_model(self, model):
		self.model = model
		self.tf_img = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image_tensor')
		self.tf_img_op = tf.summary.image('Image', self.tf_img)
		self.sess = K.get_session()
		
	def on_epoch_end(self, epoch, logs=None):
		if self.writer == None: ### Set writer value here because exp may not be defined during init.
			self.writer = tf.summary.FileWriter(self.exp.tb_dir, filename_suffix="_image")
		### Run every self.interval epochs
		if epoch % self.interval == 0:
			val_samples = self.validation_data[0]
			val_labels    = self.validation_data[1]
			val_pred_as_label_idx = val_labels.argmax(axis=1)
			# Get n samples from each predicted class
			list_of_sample_arrays = []
			for i, lbl in enumerate(self.exp.label_names):
				class_samples = val_samples[val_pred_as_label_idx == i]
				# Shuffle since samples are ordered by datasource, then select first n samples
				np.random.shuffle(class_samples)
				class_samples = [class_samples[int(len(class_samples)*random())]]
				list_of_sample_arrays.append(class_samples)

			# The samples will be in the order given by self.label_names
			for i, sample in enumerate(list_of_sample_arrays):
				_sample = np.array(sample).copy()
				figure = convert_array_to_figure(_sample, self.exp.label_names[i])
				data = fig2data(figure)
				data = np.expand_dims(data, 0)
				cm_summary = self.sess.run(self.tf_img_op, feed_dict={self.tf_img: data})
				self.writer.add_summary(cm_summary, epoch)
				plt.close(figure)
		self.writer.flush()

