import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from skimage.transform import resize

import model
from training import lr_schedule


def random_crop(img, label):
    combined = tf.concat([img, label], axis=2)
    crop = tf.image.central_crop(combined, [224, 224, 4])

    return crop[:, :, 0:3], crop[:, :, 3]


def log_images(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = keras_model.predict(train_ds.take(1))

    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image("Input", img_arr, step=epoch)
        output = np.expand_dims(np.argmax(test_pred_raw, axis=2), axis=-1)
        tf.summary.image("Predictions", output, step=epoch)


# TODO: change to open source dataset and refactor.
input_im = "./input.jpg"
label_mask = "./label.png"

img = Image.open(input_im)
img_arr = np.array(img)
img_arr = resize(img_arr, (128, 128), mode='constant', preserve_range=True)
img_arr = np.expand_dims(img_arr, axis=0)

mask = np.zeros((128, 128), dtype=np.bool)
label = Image.open(label_mask)
label_arr = np.array(label)
label_arr[label_arr == 255] = 0
label_arr = resize(label_arr.astype(np.float32), (128, 128), mode='constant', preserve_range=True)
label_arr = np.maximum(mask, label_arr)
label_arr = np.expand_dims(label_arr, axis=0)
label_arr = np.expand_dims(label_arr, axis=-1)

train_ds = tf.data.Dataset.from_tensor_slices(
    (img_arr, label_arr)).shuffle(1).batch(1)

# Model.
keras_model = model.uncompiled_unet()

# Compile model.
optimizer = tf.keras.optimizers.Adam()
keras_model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy())
keras_model.summary()

# Tensorboard callback.
log_dir = Path("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(str(log_dir.absolute()) + "\metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

# Learning rate schedule callback.
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

keras_model.fit(x=train_ds.take(1),  epochs=300, callbacks=[tensorboard_callback, lr_callback])

# Prediction
out = np.argmax(keras_model.predict(train_ds.take(1)), axis=3)
plt.imshow(out[0,:,:])
