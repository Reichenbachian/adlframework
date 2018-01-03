'''
Contains any custom metrics that aren't available in keras.
'''
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)