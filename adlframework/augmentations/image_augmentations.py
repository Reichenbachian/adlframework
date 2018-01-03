import numpy as np
from imgaug import augmenters as iaa


def imgaug_augment(sample, sequence):
	"""
	Wrapper around the imgaug library. An example sequence might be similar to as follows.
	```
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq_1 = iaa.Sequential(
	    [
	        # apply the following augmenters to most images
	        iaa.Fliplr(0.5), # horizontally flip 50% of all images
	        iaa.Flipud(0.2), # vertically flip 20% of all images
	        # crop images by -5% to 10% of their height/width
	    ],
	    random_order=True
	)
	```
	"""
	data, label = sample
	return sequence.augment_image(data), label