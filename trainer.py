import itertools

def make_batches(num_batches, batch_size,
             real_x,
             fake_in, target_good, target_bad):
        for i in range(num_batches):
            yield ((next(real_x), next(target_good)),
                   (next(fake_in), next(target_good), next(target_bad)))

def make_chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in itertools.islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk

class TrainerCallback:        
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, params, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class ProgressBarCallback(TrainerCallback):
    def on_train_begin(self, params, logs=None):
        self.num_batches = params['num_batches']
        self.num_epochs = params['epochs']
        self.current_iteration = 0;

    def on_batch_begin(self, batch, logs=None):
        self.printProgressBar(self.current_iteration, self.num_batches * self.num_epochs)

    def on_batch_end(self, batch, logs=None):
        self.current_iteration = self.current_iteration + 1
        self.printProgressBar(self.current_iteration, self.num_batches * self.num_epochs)


    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1,
                      length = 100, empty = '-', fill = '█', lend='[', rend=']'):
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
    
class Trainer:
    def train(self, train_data, test_data, params):
        pass
    
    def train_iteration(self, batch):
        pass

    
