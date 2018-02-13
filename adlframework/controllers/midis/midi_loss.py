import theano
import theano.tensor as K

epsilon = 1.0e-9
def musical_loss(y_true, y_pred):
    '''Just another crossentropy'''
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = K.nnet.categorical_crossentropy(y_pred, y_true)
    return cce