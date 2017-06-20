import tensorflow as tf
sess = tf.Session()

import math

from keras import backend as K
K.set_session(sess)

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten, LeakyReLU, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.losses import categorical_crossentropy

import keras.initializers

import numpy as np
from PIL import Image


initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
regularizer = keras.regularizers.l1_l2(1e-5, 1e-5)

# Both the generator and discriminator networks
# are lumped into one
# depending which one we are working on, different
# layers are trainable

latent_dim = 100
full_input_dim = latent_dim
h_dim = 1024
img_shape = (28, 28)
img_len = 28 * 28

g_h1 = Dense(h_dim // 4, input_dim=full_input_dim, name = "gen_h1",
             kernel_initializer=initializer, kernel_regularizer=regularizer)
g_h1a = LeakyReLU(0.2)

g_h2 = Dense(h_dim // 2, name = "gen_h2",
             kernel_initializer=initializer, kernel_regularizer=regularizer)
g_h2a = LeakyReLU(0.2)

g_h3 = Dense(h_dim, name = "gen_h3",
             kernel_initializer=initializer, kernel_regularizer=regularizer)
g_h3a = LeakyReLU(0.2)

g_y = Dense(img_len, activation="sigmoid", name="gen_y",
             kernel_initializer=initializer, kernel_regularizer=regularizer)


d_h1 = Dense(h_dim, input_dim=img_len, name='disc_h1',
             kernel_initializer=initializer, kernel_regularizer=regularizer)
d_h1a = LeakyReLU(0.2)

d_h2 = Dense(h_dim // 2, name='disc_h2',
             kernel_initializer=initializer, kernel_regularizer=regularizer)
d_h2a = LeakyReLU(0.2)

d_h3 = Dense(h_dim // 4, name='disc_h3',
             kernel_initializer=initializer, kernel_regularizer=regularizer)
d_h3a = LeakyReLU(0.2)

d_y = Dense(1, activation="sigmoid", name="disc_y")

# The genrator network only includes the generator layers
# it is never used in the actual training process
# so we don't need to compile it

generator = Sequential()
generator.add(g_h1)
generator.add(g_h1a)
generator.add(g_h2)
generator.add(g_h2a)
generator.add(g_h3)
generator.add(g_h3a)
generator.add(g_y)

# Now freeze the generator layers and make the discriminator layers trainable
g_h1.trainable = False; g_h2.trainable = False; g_h3.trainable = False; g_y.trainable = False;
d_h1.trainable = True; d_h2.trainable = True; d_h3.trainable = True; d_y.trainable = True;

# The discriminator network includes only the discriminator and
# is used for training on real-world data

discriminator = Sequential()
discriminator.add(d_h1)
discriminator.add(d_h1a)
discriminator.add(d_h2)
discriminator.add(d_h2a)
discriminator.add(d_h3)
discriminator.add(d_h3a)
discriminator.add(d_y)

discriminator.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# The generator full network will pass through the
# generator and the discriminator and minimize the
# cross entropy between the input and output
# holding the discrimiantor network constant

# hold the discriminator network constant, generator free
g_h1.trainable = True; g_h2.trainable = True; g_h3.trainable = True; g_y.trainable = True;
d_h1.trainable = False; d_h2.trainable = False; d_h3.trainable = False; d_y.trainable = False;

generator_full = Sequential()
generator_full.add(generator)
generator_full.add(discriminator)
generator_full.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# discrimiantor_full network has both the generator and discriminator
# but only trains the discriminator part to work against the generator

# Freeze the generator layers and make the discriminator layers trainable again
g_h1.trainable = False; g_h2.trainable = False; g_h3.trainable = False; g_y.trainable = False;
d_h1.trainable = True; d_h2.trainable = True; d_h3.trainable = True; d_y.trainable = True;

discriminator_full = Sequential()
discriminator_full.add(generator)
discriminator_full.add(discriminator)
discriminator_full.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Now train!

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Process the data a bit
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

print("Discrimiantor standalone:")
discriminator.summary()

print("Generator standalone:")
generator.summary()

print("Discriminator full:")
discriminator_full.summary()

print("Generator full:")
generator_full.summary()

print("Training")

TRAIN_PARAMS = {
    'batch_size': 128,
    'epochs': 100
}

# Some data generators
def fake_in_gen():
    while True:
        r = np.random.normal(size=(latent_dim))
        yield r

def good_target_gen():
    while True:
        yield np.ones(1)

def bad_target_gen():
    while True:
        yield np.zeros(1)
    
            
from gan import GANTrainer
from trainer import ProgressBarCallback
from trainer import make_chunks
        
with sess.as_default():
    progressbar = ProgressBarCallback()

    trainer = GANTrainer((fake_in_gen(), good_target_gen(), bad_target_gen()),
                         discriminator, generator,
                         discriminator_full, generator_full,
                         callbacks= [progressbar] )
    trainer.train(x_train, TRAIN_PARAMS)

print("Evaluating...")
    
with sess.as_default():
    print("Saving images")
    # Generate some content
    input = np.array(list(next(make_chunks(fake_in_gen(), 100))))
    y = generator.predict(input, batch_size=100)
    y = y.reshape((100, 28, 28))

    i = 0;
    for result in y:
        img = result * 255
        im = Image.fromarray(img)
        im = im.convert('RGB')
        im.save("%02d.jpeg" % i)
        i = i + 1
