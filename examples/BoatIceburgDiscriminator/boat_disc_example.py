from functools import partial
import pdb
import os
from imgaug import augmenters as iaa
### Data
from adlframework.retrievals.JsonFile import JsonFile
from adlframework.datasource import DataSource
from adlframework.dataentity.image_de import ImageFileDataEntity
### Model
from adlframework.nets.image_nets import medium_model
from keras.optimizers import Adadelta
from keras.losses import KLD
from adlframework.experiment import SimpleExperiment
### Controllers
from adlframework.processors.general_processors import reshape, make_categorical
from adlframework.filters.general_filters import accept_label, ignore_label
from adlframework.augmentations.image_augmentations import imgaug_augment
### Callbacks
from keras.callbacks import ModelCheckpoint
from adlframework.callbacks.image_callbacks import SaveValImages

abs_path = os.path.dirname(os.getcwd()+'/local_cache/')+'/'

train_retrieval = JsonFile(fp=abs_path+'train.json',
                            data_columns=['band_1', 'band_2'],
                            label_columns=['is_iceberg'])

val_retrieval = JsonFile(fp=abs_path+'val.json',
                            data_columns=['band_1', 'band_2'],
                            label_columns=['is_iceberg'])

## Model
net = medium_model(input_shape=(90, 125, 1),
					target_shape=(1))

### Augmentation
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
    ],
    random_order=True
)

## Controllers
controllers = [partial(reshape, shape=(90, 125, 1)),
				partial(imgaug_augment, sequence=seq)]

#### TRAINING DATASOURCE
iceberg_trainds = DataSource(train_retrieval, ImageFileDataEntity,
                  controllers = controllers + [partial(accept_label, labelnames="is_iceberg")])

boat_trainds = DataSource(train_retrieval, ImageFileDataEntity,
                  controllers = controllers + [partial(ignore_label, labelnames="is_iceberg")])

#### VALIDATION DATASOURCE
iceberg_valds = DataSource(val_retrieval, ImageFileDataEntity,
                  controllers = controllers + [partial(accept_label, labelnames="is_iceberg")])

boat_valds = DataSource(val_retrieval, ImageFileDataEntity,
                  controllers = controllers + [partial(ignore_label, labelnames="is_iceberg")])

### Combining DataSources
train_ds = iceberg_trainds + boat_trainds
val_ds = iceberg_valds + boat_valds


#### Callbacks
callbacks = [ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')]

#### Define experiment
exp = SimpleExperiment(train_datasource=train_ds,
						validation_datasource=val_ds,
						loss=KLD,
						optimizer=Adadelta(),
						network = net,
						callbacks=callbacks)

exp.run()