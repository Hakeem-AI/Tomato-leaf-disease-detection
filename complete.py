from keras.models import Model
import keras
import  tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, BatchNormalization, Dense, Flatten
from keras.layers import Activation, MaxPool2D, Rescaling
import numpy as np
import keras.backend as k
k.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)
         ])
    logical_gpus = tf.config.list_logical_devices('GPU')

    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
imagesize = (180, 180)
trainds, val = tf.keras.preprocessing.image_dataset_from_directory(
    directory='train',
    validation_split=0.2,
    seed=1377,
    shuffle=True,
    subset='both',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(180, 180))
train_ds = trainds.prefetch(tf.data.AUTOTUNE)
val_ds = val.prefetch(tf.data.AUTOTUNE)
def make_model(input_shape, num_classes):
    inputs = Input(input_shape)
    inputs = Rescaling(1/255)(inputs)
    x = inputs
    x = Conv2D(16, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPool2D((2,2))(x)

    x = Conv2D(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, 7, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, 3, strides=2, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, 3, padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation= 'relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model1 = Model(inputs,outputs)
    return model1
model = make_model(imagesize + (3,), num_classes=11)
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds,batch_size=10, epochs=50, callbacks=callbacks, validation_data=val_ds)

np.save('history.npy', history.history)
model.save("save.h5")