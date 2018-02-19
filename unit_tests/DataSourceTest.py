from functools import partial
### Data
from adlframework.retrievals.MNIST import MNIST_retrieval
from adlframework.datasource import DataSource
from adlframework.dataentity.image_de import ImageFileDataEntity
from tqdm import tqdm

### Controllers
processors = [partial(reshape, out_shape=(28, 28, 1)),
			  partial(make_categorical, num_classes=10)]

### Load Data
mnist_retrieval = MNIST_retrieval()
mnist_ds = DataSource(mnist_retrieval, ImageFileDataEntity, processors=processors, workers=4)

train_ds, temp = DataSource.split(mnist_ds, split_percent=.6)
val_ds, test_ds = DataSource.split(temp, split_percent=.6)

### Load network
net = mnist_test(input_shape=(28, 28, 1), target_shape=10)

for i in tqdm(net):
	pass