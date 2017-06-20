from kerasvis import DBLogger

import itertools
import numpy as np


def _batches(num_batches, batch_size,
             real_x, real_y,
             fake_in, fake_good_out, fake_bad_out):
        for i in range(num_batches):
            yield ((next(real_x), next(real_y)),
                   (next(fake_in), next(fake_good_out), next(fake_bad_out)))

def _chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in itertools.islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk

class TrainerCallback:
    def set_params(self, params):
        self.params = params
        
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Trainer:
    def train(self, train_data, batch_size=128, epochs=5):
        pass
    
    def train_iteration(self, batch):
        pass

    
class GANTrainer(Trainer):
    def __init__(self, data_generators,
                 discriminator, generator, disc_full, gen_full, callbacks=()):
        self.data_generators = data_generators
        self.discriminator = discriminator
        self.generator = generator
        self.discriminator_full = disc_full
        self.generator_full = gen_full

        self.callbacks = callbacks

    def train(self, train_data, batch_size=128, epochs=5):
        map(lambda x: x.on_train_begin(), self.callbacks)
        
        # Train_data contains real_x, real_y
        real_x = train_data[0]
        real_y = train_data[1]

        # Make batches
        num_batches = int(real_x.shape[0] / batch_size)

        for epoch in range(epochs):
            batches = _batches(num_batches, batch_size,
                               _chunks(real_x, batch_size),
                               _chunks(real_y, batch_size),
                               _chunks(self.data_generators[0], batch_size),
                               _chunks(self.data_generators[1], batch_size),
                               _chunks(self.data_generators[2], batch_size))
            for batch in batches:
                self.train_iteration(batch)
        #self.discriminator.train_on_batch(real_x, real_y_aug)
        
        map(lambda x: x.on_train_end(), self.callbacks)
        
    def train_iteration(self, batch):
        real_x = np.array(list(batch[0][0]))
        real_y = np.array(list(batch[0][1]))
        
        fake_in = np.array(list(batch[1][0]))
        fake_good_out = np.array(list(batch[1][1]))
        fake_bad_out = np.array(list(batch[1][2]))

        # Train the discriminator on the real x and real y
        self.discriminator.train_on_batch(real_x, real_y)

        # Train the generator in the generator-discriminator stack
        self.generator_full.train_on_batch(fake_in, fake_good_out)

        # Train the discriminator in the generator-discriminator stack
        self.discriminator_full.train_on_batch(fake_in, fake_bad_out)
