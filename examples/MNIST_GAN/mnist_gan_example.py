'''
Author: Alex Reichenbach
References: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py
January 2, 2018
'''

from functools import partial
from tqdm import tqdm
import numpy as np
# Data
from adlframework.retrievals.MNIST import MNIST_retrieval
from adlframework.datasource import DataSource
from adlframework.dataentity.image_de import ImageFileDataEntity
from adlframework.dataentity.noise_de import NoiseDataEntity
# Model
from mnist_gan_network import mnist_gan_network
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from adlframework.experiment import AdvancedExperiment
# Controllers
from adlframework.controllers.general import reshape, make_categorical, to_np_arr
# Callbacks
from adlframework.controllers.images import SaveValImages

import pdb

# Consts
batch_size = 100
ds_vars = {'batch_size': batch_size}
randomDim = 100

# Load Data
mnist_retrieval = MNIST_retrieval()
mnist_ds = DataSource(mnist_retrieval, ImageFileDataEntity, **ds_vars)
noise_ds = DataSource(None, NoiseDataEntity, shape=randomDim, **ds_vars)

train_ds, temp = DataSource.split(mnist_ds, split_percent=.6)
val_ds, test_ds = DataSource.split(temp, split_percent=.6)

# Load network
net = mnist_gan_network(randomDim=randomDim)

# Callbacks
callbacks = []


def epoch():
    gloss = []
    dloss = []
    for _ in tqdm(xrange(batch_size)):
        # Get a random set of input noise and images
        noise = noise_ds.next()[0]  # select the X and ignore the y
        imageBatch = mnist_ds.next()[0]

        # Generate fake MNIST images
        generatedImages = net.generator.predict(noise)
        X = np.concatenate([imageBatch, generatedImages])

        # Labels for generated and real data
        yDis = np.zeros(2*batch_size)
        yDis[:batch_size] = 0.9  # One-sided label smoothing

        # Train discriminator
        net.discriminator.trainable = True
        dloss.append(net.discriminator.train_on_batch(X, yDis))

        # Train generator
        noise = noise_ds.next()[0]
        yGen = np.ones(batch_size)
        net.discriminator.trainable = False
        gloss.append(net.gan.train_on_batch(noise, yGen))

    return {"dloss": np.mean(dloss), "gloss": np.mean(gloss)}

# Create and run experiment
exp = AdvancedExperiment(epoch=epoch,
                         network=net,
                         callbacks=callbacks,
                         epochs=10)
exp.run()
