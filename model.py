import sys
sys.setrecursionlimit(10000)
import os
os.environ['THEANO_FLAGS'] = "device=gpu1"
os.environ['KERAS_BACKEND']='theano'
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers.core import Flatten, Dense, Dropout

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

dropOut=0.25
H=32
W=32

input_img = Input(shape=(1, H, W))

x = Convolution2D(48, 5, 5, activation='relu', border_mode='same',subsample=(2, 2))(input_img)
x = Dropout(dropOut)(x)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)



x = Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(2, 2))(x)
x = Dropout(dropOut)(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)



x = Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(2, 2))(x)
x = Dropout(dropOut)(x)

x= Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)



from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D

x= UpSampling2D(size=(2, 2))(x)
x= Convolution2D(256, 4, 4, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)

x= Convolution2D(256, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)
x= Convolution2D(128, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)




x= UpSampling2D(size=(2, 2))(x)
x= Convolution2D(128, 4, 4, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)

x= Convolution2D(128, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)
x= Convolution2D(48, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)




x= UpSampling2D(size=(2, 2))(x)
x= Convolution2D(48, 4, 4, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)

x= Convolution2D(28, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
x = Dropout(dropOut)(x)
decoded= Convolution2D(1, 3, 3, activation='relu', border_mode='same',subsample=(1, 1))(x)
from keras.optimizers import Adam
adam = Adam(lr=1e-4)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
autoencoder.summary()

## Import data into X_train and X_test
autoencoder.fit(X_train, X_train,
        shuffle=True,
        nb_epoch=1000,
        batch_size=50, verbose=1,
        validation_data=(X_test, X_test))


