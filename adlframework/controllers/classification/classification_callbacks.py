"""
Only callbacks specific to classification should be in here.
"""
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
import numpy as np
from adlframework.utils import fig2data
import matplotlib.pyplot as plt


# METIRCS
def recall_per_class(y_true, y_pred, cls):
    """ Calculate recall for the given class index: cls
    y_true/pred is a Tensor with shape (samples in batch, output)
    Arguments:
        y_true : 1/0
        y_pred : probability
        cls    : class index to get recall for
    Returns :
        Recall for the selected class"""
    with tf.name_scope("recall_per_class"):
        # Pick selected class outputs only
        y_true = y_true[:, cls]
        y_pred = y_pred[:, cls]
        # Threshold prediction probability
        y_pred = y_pred >= tf.constant(0.5)
        # Cast bools to 1/0 int
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        
        # Find tp, fp
        tp = tf.count_nonzero(   y_pred  * y_true) # True prediction and True label
        fn = tf.count_nonzero((1-y_pred) * y_true) # False prediction but True label 
        
        # convert to float
        tp = tf.cast(tp, tf.float32)
        fn = tf.cast(fn, tf.float32)
        
        recall = tp / (tp+fn)
        return recall
    
def precision_per_class(y_true, y_pred, cls):
    """ Calculate precision for the given class index: cls
    y_true/pred is a Tensor with shape (samples in batch, output)
    Arguments:
        y_true : 1 or 0
        y_pred : probability
        cls    : class index to get recall for
    Returns :
        precision for the selected class"""
    with tf.name_scope("recall_per_class"):
        # Pick selected class outputs only
        y_true = y_true[:, cls]
        y_pred = y_pred[:, cls]
        # Threshold prediction probability
        y_pred = y_pred >= tf.constant(0.5)
        # Cast bools to 1/0 int
        y_pred = tf.cast(y_pred, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        
        # Find tp, fp
        tp = tf.count_nonzero(   y_pred  * y_true) # True prediction and True label 
        fp = tf.count_nonzero(   y_pred  * (1-y_true)) #True pred but False label
        # convert to float
        tp = tf.cast(tp, tf.float32)
        fp = tf.cast(fp, tf.float32)
        
        precision = tp / (tp+fp)
        return precision


def num_per_class(y_true, y_pred, cls):
    """
    Returns the number of samples in a specific class.
    Assumes that y_true contains 1 or 0.
    """
    return tf.count_nonzero(tf.cast(y_true, tf.int32))

def pred_per_class(y_true, y_pred, cls):
    return tf.count_nonzero(tf.cast(y_pred >= tf.constant(0.5), tf.int32))

class LabelHistogram(Callback):
    """
    Parameters:
    ----------
    log_dir:    Tensorboard directory to save the event files
    interval:   Interval (number of epochs) between consecutive roc analysis
    label_names: Names of labels in the order of network outputs
    sample_rate: Sampling rate of the input audio
    samples_per_class : The number of samples per class to be added to Tensorboard at each epoch.
                        The samples will be ordered by their predicted class.
    Example Usage:
        PlayAudio('results/', ['angry','neutral', 'other'], 16000)
    """

    def __init__(self, log_dir, label_names, interval = 1):
        self.writer      = tf.summary.FileWriter(log_dir, filename_suffix="_hist")
        self.label_names = label_names
        self.interval    = interval
        
    def set_model(self, model):
        self.model = model
        self.tf_img = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='histogram_tensor')
        self.tf_img_op = tf.summary.image('Label Histogram', self.tf_img)
        self.sess = K.get_session()
        
    def on_epoch_end(self, epoch, logs=None):
        ## TO-DO: Add x-axis labels.
        if epoch % self.interval == 0:
            labels = self.validation_data[1].argmax(axis=1)
            figure = plt.figure()
            plt.hist(labels, bins=len(np.unique(labels)))
            plt.title("Label Histogram in Validation")
            data = fig2data(figure)
            data = np.expand_dims(data, 0)
            cm_summary = self.sess.run(self.tf_img_op, feed_dict={self.tf_img: data})
            self.writer.add_summary(cm_summary, epoch)
            self.writer.flush()
            plt.close(figure)

