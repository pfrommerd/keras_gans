import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.losses import categorical_crossentropy

import keras.initializers

import numpy as np
from PIL import Image


initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

# Both the generator and discriminator networks
# are lumped into one
# depending which one we are working on, different
# layers are trainable

g_h0 = Dense(8000, input_dim=10, activation="relu",
             kernel_initializer=initializer, name="gen_h0")
g_h1 = Dense(8000, activation="relu",
             kernel_initializer=initializer, name="gen_h1")
g_y = Dense(784, activation="sigmoid",
             kernel_initializer=initializer, name="gen_y")
d_h0 = Dense(128, input_dim=784, activation="relu",
             kernel_initializer=initializer, name='disc_h0')
d_h1 = Dense(128, activation="relu",
             kernel_initializer=initializer, name="disc_h1")
d_y = Dense(11, activation="softmax", name="disc_y")

# The genrator network only includes the generator layers
# it is never used in the actual training process
# so we don't need to compile it

generator = Sequential()
generator.add(g_h0)
generator.add(g_h1)
generator.add(g_y)

# Now freeze the generator layers and make the discriminator layers trainable
g_h0.trainable = False; g_h1.trainable = False; g_y.trainable = False;
d_h0.trainable = True; d_h1.trainable = True; d_y.trainable = True;

# The discriminator network includes only the discriminator and
# is used for training on real-world data

discriminator = Sequential()
discriminator.add(d_h0)
discriminator.add(d_h1)
discriminator.add(d_y)
discriminator.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# The generator full network will pass through the
# generator and the discriminator and minimize the
# cross entropy between the input and output
# holding the discrimiantor network constant

# hold the discriminator network constant, generator free
g_h0.trainable = True; g_h1.trainable = True; g_y.trainable = True;
d_h0.trainable = False; d_h1.trainable = False; d_y.trainable = False;

generator_full = Sequential()
generator_full.add(generator)
generator_full.add(discriminator)
generator_full.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# discrimiantor_full network has both the generator and discriminator
# but only trains the discriminator part to work against the generator

# Freeze the generator layers and make the discriminator layers trainable again
g_h0.trainable = False; g_h1.trainable = False; g_y.trainable = False;
d_h0.trainable = True; d_h1.trainable = True; d_y.trainable = True;

def negative_crossentropy(y_true, y_pred):
    return -categorical_crossentropy(y_true, y_pred)

discriminator_full = Sequential()
discriminator_full.add(generator)
discriminator_full.add(discriminator)
#discriminator_full.compile(optimizer='rmsprop', loss=negative_crossentropy, metrics=['accuracy'])
discriminator_full.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Now train!

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Process the data a bit
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

y_train_aug = np.hstack((y_train,
                         np.zeros((y_train.shape[0], 1), dtype=y_train.dtype)))
y_test_aug = np.hstack((y_test,
                         np.zeros((y_test.shape[0], 1), dtype=y_test.dtype)))


print("Discrimiantor standalone:")
discriminator.summary()

print("Generator standalone:")
generator.summary()

print("Discriminator full:")
discriminator_full.summary()

print("Generator full:")
generator_full.summary()

print("Training")

ALL_CATEGORIES = np.identity(10) # Will contain all possible categories to input into the generator
ALL_CATEGORIES_AUG = np.hstack((ALL_CATEGORIES, np.zeros((ALL_CATEGORIES.shape[0], 1), dtype=ALL_CATEGORIES.dtype)))
FAKE_CATEGORY = np.array([0,0,0,0,0,0,0,0,0,0,1])

# Some data generators
def fake_in_gen():
    while True:
        for i in range(ALL_CATEGORIES.shape[0]):
            yield ALL_CATEGORIES[i]

def fake_good_out_gen():
    while True:
        for i in range(ALL_CATEGORIES_AUG.shape[0]):
            yield ALL_CATEGORIES_AUG[i]

def fake_bad_out_gen():
    while True:
        yield FAKE_CATEGORY
    
            
from trainer import GANTrainer
        
with sess.as_default():
    trainer = GANTrainer((fake_in_gen(), fake_good_out_gen(), fake_bad_out_gen()),
                         discriminator, generator,
                         discriminator_full, generator_full)
    trainer.train((x_train, y_train_aug))
