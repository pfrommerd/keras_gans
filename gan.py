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

print("Discrimiantor standalone:")
discriminator.summary()

print("Generator standalone:")
generator.summary()

print("Discriminator full:")
discriminator_full.summary()

print("Generator full:")
generator_full.summary()

print("Training")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1,
                      length = 100, empty = '-', fill = '█', lend='|', rend='|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + empty * (length - filledLength)
    print('\r%s %s%s%s %s%% %s' % (prefix, lend, bar, rend, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
        
BATCH_SIZE = 128
ITERATIONS = 500
ALL_CATEGORIES = np.identity(10) # Will contain all possible categories to input into the generator
FAKE_CATEGORY = np.array([[0,0,0,0,0,0,0,0,0,0,1]])

with sess.as_default():
    for i in range(ITERATIONS):
        # Train just the discriminator on the real images
        real_indices = np.random.randint(0, x_train.shape[0], size=BATCH_SIZE)
        real_x = x_train[real_indices, :]
        real_y = y_train[real_indices, :]
        # Add a zero in the fake slot
        
        discriminator.train_on_batch(real_x, real_y_aug)

        # Train the discriminator and generator
        # on the full stack (i.e with images outputed from the generator)
        # Generate some fake numbers by sampling from the all categories
        #fake_gen_in = ALL_CATEGORIES[np.random.randint(0, ALL_CATEGORIES.shape[0], size=BATCH_SIZE), :]
        fake_gen_in = ALL_CATEGORIES
        good_disc_out = np.hstack((fake_gen_in, np.zeros((fake_gen_in.shape[0], 1), dtype=fake_gen_in.dtype)))
        fake_disc_out = np.repeat(FAKE_CATEGORY, fake_gen_in.shape[0], axis=0)

        #fake_y = np.random.rand(BATCH_SIZE, 10)
        #np.apply_along_axis(softmax, 0, fake_y)
        #fake_y = ALL_CATEGORIES
        

        for k in range(1):
            # Train the full generator-discriminator stack from the discriminator side (classify fake as fake)
            discriminator_full.train_on_batch(fake_gen_in, fake_disc_out)
            # Train the full generator-discrminator stack from the generator side
            generator_full.train_on_batch(fake_gen_in, good_disc_out)
        printProgressBar(i, ITERATIONS, prefix='Progress:', suffix= 'Complete', empty=' ', fill='%', lend='[', rend=']', length=50)

print("Done!")

# Evaluate the model
y_test_aug = np.hstack((y_test, np.zeros((y_test.shape[0], 1), dtype=y_test.dtype)))

with sess.as_default():
    print(discriminator.evaluate(x_test, y_test_aug))
    # Generate some content
    y = generator.predict(ALL_CATEGORIES, batch_size=10)
    y = y.reshape((10, 28, 28))

    i = 0;
    for result in y:
        img = result * 255
        im = Image.fromarray(img)
        im = im.convert('RGB')
        im.save("%d.jpeg" % i)
        i = i + 1
