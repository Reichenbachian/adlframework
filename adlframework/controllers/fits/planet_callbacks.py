import io
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
import numpy as np
from random import random
import matplotlib.pyplot as plt
from adlframework.utils import fig2data

def convert_array_to_figure(data, label):
    figure = plt.figure()
    plt.plot(data)
    plt.title(label)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    return figure

class LcGraph(Callback):
    """
    Parameters:
    ----------
    log_dir:    Tensorboard directory to save the event files
    label_names: Names of labels in the order of network outputs
    interval : How often a light curve will be saved.
    """

    def __init__(self, log_dir, label_names, interval = 1):
        self.writer      = tf.summary.FileWriter(log_dir, filename_suffix="_lcurve")
        self.label_names = label_names
        self.interval    = interval
        self.log_dir = log_dir
        
    def set_model(self, model):
        self.model = model
        self.tf_img = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='lcurve_tensor')
        self.tf_img_op = tf.summary.image('LightCurve', self.tf_img)
        self.sess = K.get_session()
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.interval == 0:
            val_samples = self.validation_data[0]
            val_labels    = self.validation_data[1] # probabilitles shape (X, 3)
            val_pred_as_label_idx = val_labels.argmax(axis=1) # index of predicted label: shape (X)
            # Get n samples from each predicted class
            list_of_sample_arrays = []
            for i, lbl in enumerate(self.label_names):
                class_samples = val_samples[val_pred_as_label_idx == i]
                # Shuffle since samples are ordered by datasource, then select first n samples
                np.random.shuffle(class_samples)
                class_samples = [class_samples[int(len(class_samples)*random())]]
                list_of_sample_arrays.append(class_samples)

            # The samples will be in the order given by self.label_names
            for i, sample in enumerate(list_of_sample_arrays):
                _sample = np.array(sample).copy().flatten()
                figure = convert_array_to_figure(_sample, self.label_names[i])
                data = fig2data(figure)
                data = np.expand_dims(data, 0)
                cm_summary = self.sess.run(self.tf_img_op, feed_dict={self.tf_img: data})
                self.writer.add_summary(cm_summary, epoch)
                plt.close(figure)
        self.writer.flush()