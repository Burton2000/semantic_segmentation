import tensorflow as tf
from tensorflow.keras import layers

def uncompiled_unet():
    inputs = tf.keras.Input((128, 128, 3))
    layers.Lambda(lambda x: x / 255)(inputs)
    conv1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1) # 112

    conv2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2) # 56

    conv3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3) # 28

    conv4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4)

    up6 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3) #56
    conv6 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    up7 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv2], axis=3) #112
    conv7 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.concatenate([layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), conv1], axis=3) #224
    conv8 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    conv10 = layers.Conv2D(2, (1, 1), activation='sigmoid')(conv8)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    return model
