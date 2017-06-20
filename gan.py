import numpy as np

from trainer import Trainer
from trainer import make_batches
from trainer import make_chunks

from kerasvis import DBLogger

class GANTrainer(Trainer):
    def __init__(self, data_generators,
                 discriminator, generator, disc_full, gen_full, callbacks=()):
        self.data_generators = data_generators
        self.discriminator = discriminator
        self.generator = generator
        self.discriminator_full = disc_full
        self.generator_full = gen_full

        self.callbacks = callbacks

    def train(self, train_data, params):
        
        # Train_data contains real_x, real_y
        real_x = train_data[0]
        real_y = train_data[1]

        # Make batches
        batch_size = params['batch_size']
        num_batches = int(real_x.shape[0] / batch_size)
        num_batches = 5
        params['num_batches'] = num_batches
        epochs = params['epochs']

        for x in self.callbacks: x.on_train_begin(params)

        for epoch in range(epochs):
            for x in self.callbacks: x.on_epoch_begin(epoch)
            batches = make_batches(num_batches, batch_size,
                               make_chunks(real_x, batch_size),
                               make_chunks(real_y, batch_size),
                               make_chunks(self.data_generators[0], batch_size),
                               make_chunks(self.data_generators[1], batch_size),
                               make_chunks(self.data_generators[2], batch_size))
            for batch in batches:
                self.train_iteration(batch)
            for x in self.callbacks: x.on_epoch_end(epoch)
        #self.discriminator.train_on_batch(real_x, real_y_aug)
        
        for x in self.callbacks: x.on_train_end()

        
    def train_iteration(self, batch):
        for x in self.callbacks: x.on_batch_begin(batch)
        
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
        
        for x in self.callbacks: x.on_batch_end(batch)

