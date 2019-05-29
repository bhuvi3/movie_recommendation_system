import keras

from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.layers import Flatten, Reshape
from keras.models import Model, Sequential


def get_model():
    autoencoder = Sequential()
    autoencoder.add(Conv2D(64, (7, 7), padding='same', input_shape=(512, 512, 3)))  # first conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (5, 5), padding='same'))  # second conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(256, (3, 3), padding='same'))  # third conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(256, (3, 3), padding='same'))  # fourth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), padding='same'))  # fifth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(Conv2D(4, (1, 1), padding='same'))  # sixth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Flatten())  # resulting size: (None, 1024)
    autoencoder.add(Reshape((16, 16, 4)))
    autoencoder.add(Conv2D(4, (1, 1), padding='same'))  # -sixth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), padding='same'))  # -fifth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(256, (3, 3), padding='same'))  # -fourth conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(256, (3, 3), padding='same'))  # -third conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (5, 5), padding='same'))  # -second conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (7, 7), padding='same'))  # -first conv layer
    autoencoder.add(BatchNormalization())
    autoencoder.add(LeakyReLU(alpha=0.1))
    # 1x1 conv to reduce channels to bring into input image shape
    autoencoder.add(Conv2D(3, (1, 1), padding='same', activation='relu'))
    # might need a reshape layer for enabling image comparison
    autoencoder.summary()
    return autoencoder, autoencoder.get_layer('flatten_1').output
