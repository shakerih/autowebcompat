import argparse
import random
import time

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import confusion_matrix

from autowebcompat import network
from autowebcompat import utils



def get_random_hp_config():
    optimizer_options = ['sgd', 'adam', 'adagrad', 'rms']
    hp = ['vgg16true'] # network
    hp.append(random.choice(optimizer_options)) #optimizer
    hp.append(random.uniform(0.00001, 0.01)) #learning rate
    hp.append(np.exp(random.uniform(np.log(0.001), np.log(0.5)))) #dropout1
    hp.append(np.exp(random.uniform(np.log(0.001), np.log(0.5)))) #dropout2
    hp.append(np.exp(random.uniform(np.log(0.0000001), np.log(0.0001)))) #l2 weight decay 1
    hp.append(np.exp(random.uniform(np.log(0.0000001), np.log(0.0001)))) #l2 weight decay 2
    hp.append(random.uniform(0.6, 1)) #momentum
    hp.append(random.choice([True, False])) #nesterov
    hp.append(random.uniform(0.0000001, 0.00001)) #decay
    hp.append(random.uniform(0.000000001, 0.0000001)) #epsilon
    '''
parser.add_argument('-n', '--network', type=str, choices=network.SUPPORTED_NETWORKS, help='Select the network to use for training')
parser.add_argument('-l', '--labels', type=str, default='labels.csv', help='Location of labels file to be used for training')
parser.add_argument('-o', '--optimizer', type=str, choices=network.SUPPORTED_OPTIMIZERS, default='sgd', help='Select the optimizer to use for training')
parser.add_argument('-w', '--weights', type=str, help='Location of the weights to be loaded for the given model')
parser.add_argument('-bw', '--builtin_weights', type=str, choices=network.SUPPORTED_WEIGHTS, help='Select the weights to be loaded for the given model')
parser.add_argument('-ct', '--classification_type', type=str, choices=utils.CLASSIFICATION_TYPES, default=utils.CLASSIFICATION_TYPES[0], help='Select the classification_type for training')
parser.add_argument('-es', '--early_stopping', dest='early_stopping', action='store_true', help='Stop training training when validation accuracy has stopped improving.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Increase the rate of gradient step size by increasing value.')
parser.add_argument('-do1', '--dropout1', type=float, default=0.0, help='Increase the rate of regularization by increasing percentage of nodes to drop on first FC layer')
parser.add_argument('-do2', '--dropout2', type=float, default=0.0, help='Increase the rate of regularization by increasing percentage of nodes to drop on second FC layer')
parser.add_argument('-ls1', '--l2_strength1', type=float, default=0.0, help='Increase to increase the regularization strength on first FC layer.')
parser.add_argument('-ls2', '--l2_strength2', type=float, default=0.0, help='Increase to increase the regularization strength on second FC layer.')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Increase to increase the momentum effect of stochastic optimizers.')
parser.add_argument('-nest', '--nesterov', type=bool, default=True, help='Use True to apply nesterov momentum')
parser.add_argument('-d', '--decay', type=float, default=1e-6, help='Increase to speed up the rate at which the learning rate shrinks')
parser.add_argument('-e', '--epsilon', type=float, default=None, help='Fuzz factor, for use with Adam and Adagrad optimizers')
    '''
    return hp


class HP:
  def __init__(self, network, optimizer, lr, dropout1, dropout2, l21, l22, momentum, nesterov, decay, epsilon):
    self.network = network
    self.optimizer = optimizer
    self.lr = lr
    self.dropout1 = dropout1
    self.dropout2 = dropout2
    self.l21 = l21
    self.l22 = l22
    self.momentum = momentum
    self.nesterov = nesterov
    self.decay = decay
    self.epsilon = epsilon
def run_and_loss(num_iters, hp):


    BATCH_SIZE = 32
    EPOCHS = num_iters
    random.seed(42)
    hypdict = HP(hp[0], hp[1],hp[2],hp[3], hp[4], hp[5], hp[6], hp[7], hp[8], hp[9], hp[10])

    class Timer(Callback):
        def on_train_begin(self, logs={}):
            self.train_begin_time = time.time()
            self.epoch_times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_begin_time = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.epoch_times.append(time.time() - self.epoch_begin_time)

        def on_train_end(self, logs={}):
            self.train_time = time.time() - self.train_begin_time
            print(HP)


    labels = utils.read_labels('labels.csv')

    utils.prepare_images()
    all_image_names = [i for i in utils.get_images() if i in labels]
    all_images = sum([[i + '_firefox.png', i + '_chrome.png'] for i in all_image_names], [])
    image = utils.load_image(all_images[0])
    input_shape = image.shape

    SAMPLE_SIZE = len(all_image_names)
    TRAIN_SAMPLE = 80 * (SAMPLE_SIZE // 100)
    VALIDATION_SAMPLE = 10 * (SAMPLE_SIZE // 100)
    TEST_SAMPLE = SAMPLE_SIZE - (TRAIN_SAMPLE + VALIDATION_SAMPLE)


    def load_pair(fname):
        return [fname + '_firefox.png', fname + '_chrome.png']


    random.shuffle(all_image_names)
    images_train, images_validation, images_test = all_image_names[:TRAIN_SAMPLE], all_image_names[TRAIN_SAMPLE:VALIDATION_SAMPLE + TRAIN_SAMPLE], all_image_names[SAMPLE_SIZE - TEST_SAMPLE:]


    def couples_generator(images):
        for i in images:
            yield load_pair(i), utils.to_categorical_label(labels[i], 'Y vs D + N')


    def gen_func(images):
        return couples_generator(images)


    train_couples_len = sum(1 for e in gen_func(images_train))
    validation_couples_len = sum(1 for e in gen_func(images_validation))
    test_couples_len = sum(1 for e in gen_func(images_test))

    data_gen = utils.get_ImageDataGenerator(all_images, input_shape)
    train_iterator = utils.CouplesIterator(utils.make_infinite(gen_func, images_train), input_shape, data_gen, BATCH_SIZE)
    validation_iterator = utils.CouplesIterator(utils.make_infinite(gen_func, images_validation), input_shape, data_gen, BATCH_SIZE)
    test_iterator = utils.CouplesIterator(utils.make_infinite(gen_func, images_test), input_shape, data_gen, BATCH_SIZE)

    model = network.create(input_shape, hp[0], None, \
                    None, hp[3], hp[4], \
                    hp[5], hp[6])

    network.compile(model, hp[1], hp[2], hp[9], \
            hp[7], hp[8], hp[10])

    timer = Timer()
    callbacks_list = [ModelCheckpoint('best_train_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'), timer]

    callbacks_list.append(EarlyStopping(monitor='val_accuracy', patience=2))

    train_history = model.fit_generator(train_iterator, callbacks=callbacks_list, validation_data=validation_iterator, steps_per_epoch=train_couples_len / BATCH_SIZE, validation_steps=validation_couples_len / BATCH_SIZE, epochs=EPOCHS)
    score = model.evaluate_generator(test_iterator, steps=test_couples_len / BATCH_SIZE)
    print(score)

    y_true, y_pred = [], []
    for i, (x, y) in enumerate(test_iterator):
        y_pred_batch = model.predict_on_batch(x)
        y_pred_batch = np.where(y_pred_batch < 0.5, 1, 0)
        y_true.extend(y)
        y_pred.extend(y_pred_batch.flatten().tolist())
        if i == test_couples_len // BATCH_SIZE:
            break

    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))

    train_history = train_history.history
    train_history.update({'epoch time': timer.epoch_times})
    information = vars(hypdict)
    information.update({'Accuracy': score, 'Train Time': timer.train_time, 'Number of Train Samples': train_couples_len, 'Number of Validation Samples': validation_couples_len, 'Number of Test Samples': test_couples_len})
    utils.write_train_info(information, model, train_history)

    return 1/score

max_iter = 81 #max iterations/epochs per configurations
eta = 3 #downsampling rate
logeta = lambda x: np.log(x)/np.log(eta)
s_max = int(logeta(max_iter)) #num of unique executions of successive halving minus one.
B= (s_max+1)*max_iter #num iterations (without reuse) per executions of successive halving

for s in reversed(range(s_max+1)):
    n = int(np.ceil(int(B/max_iter/(s+1))*eta*s)) #initial num of configurations
    r = max_iter*eta**(-s) #initial num iterations to run configurations for

    ### successive halving
    T = [get_random_hp_config() for i in range(n)]
    for i in range(s+1):
        n_i = n*eta**(-i)
        r_i = r*eta**(i)
        val_losses = [run_and_loss(num_iters=r_i, hp = t) for t in T]
        T = [T[i] for i in np.argsort(val_losses)[0:int(n_i/eta)]]
