from keras import backend as K
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import ActivityRegularization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
SUPPORTED_NETWORKS = ['inception', 'vgglike', 'vgg16', 'vgg16true', 'vgg19', \
                    'simnet', 'simnetlike', 'resnet50']
SUPPORTED_OPTIMIZERS = {
    'sgd': SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True),
    'adam': Adam(),
    'nadam': Nadam(),
    'rms': RMSprop(),
    'adagrad': Adagrad()
}
SUPPORTED_WEIGHTS = ['imagenet']
SUPPORTED_NETWORKS_WITH_WEIGHTS = ['vgg16', 'vgg19', 'resnet50']


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_mlp(input_shape, weights):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def create_vgg16_network(input_shape, weights):
    base_model = VGG16(input_shape=input_shape, weights=weights)
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


def create_vgg19_network(input_shape, weights):
    base_model = VGG19(input_shape=input_shape, weights=weights)
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


def create_vgglike_network(input_shape, weights):
    input = Input(shape=input_shape)

    # input: 192x256 images with 3 channels -> (192, 256, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    # x = Dense(2, activation='softmax')(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)


def create_simnet_network(input_shape, weights):
    L2_REGULARIZATION = 0.001

    input = Input(shape=input_shape)

    # CNN 1
    vgg16 = create_vgg16_network(input_shape, weights)
    cnn_1 = vgg16(input)

    # CNN 2
    # Downsample by 4:1
    cnn_2 = MaxPooling2D(pool_size=(4, 4))(input)
    cnn_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Dropout(0.5)(cnn_2)
    cnn_2 = Flatten()(cnn_2)
    cnn_2 = Dense(1024, activation='relu')(cnn_2)

    # CNN 3
    # Downsample by 8:1
    cnn_3 = MaxPooling2D(pool_size=(8, 8))(input)
    cnn_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_3)
    cnn_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_3)
    cnn_3 = Dropout(0.5)(cnn_3)
    cnn_3 = Flatten()(cnn_3)
    cnn_3 = Dense(512, activation='relu')(cnn_3)

    concat_2_3 = concatenate([cnn_2, cnn_3])
    concat_2_3 = Dense(1024, activation='relu')(concat_2_3)
    l2_reg = ActivityRegularization(l2=L2_REGULARIZATION)(concat_2_3)

    concat_1_l2 = concatenate([cnn_1, l2_reg])
    output = Dense(4096, activation='relu')(concat_1_l2)

    return Model(input, output)


def create_simnetlike_network(input_shape, weights):
    L2_REGULARIZATION = 0.005

    input = Input(shape=input_shape)

    # CNN 1
    vgg16 = create_vgglike_network(input_shape, weights)
    cnn_1 = vgg16(input)

    # CNN 2
    # Downsample by 4:1
    cnn_2 = MaxPooling2D(pool_size=(4, 4))(input)
    cnn_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(cnn_2)
    cnn_2 = Dropout(0.5)(cnn_2)
    cnn_2 = Flatten()(cnn_2)
    cnn_2 = Dense(64, activation='relu')(cnn_2)

    # CNN 3
    # Downsample by 8:1
    cnn_3 = MaxPooling2D(pool_size=(8, 8))(input)
    cnn_3 = Conv2D(16, (3, 3), padding='same', activation='relu')(cnn_3)
    cnn_3 = Conv2D(16, (3, 3), padding='same', activation='relu')(cnn_3)
    cnn_3 = Dropout(0.5)(cnn_3)
    cnn_3 = Flatten()(cnn_3)
    cnn_3 = Dense(32, activation='relu')(cnn_3)

    concat_2_3 = concatenate([cnn_2, cnn_3])
    concat_2_3 = Dense(128, activation='relu')(concat_2_3)
    l2_reg = ActivityRegularization(l2=L2_REGULARIZATION)(concat_2_3)

    concat_1_l2 = concatenate([cnn_1, l2_reg])
    output = Dense(256, activation='relu')(concat_1_l2)

    return Model(input, output)


def create_inception_network(input_shape, weights):
    """
       Simple architecture with one layer of inception model
       param input_shape: shape of the input image
    """

    input = Input(shape=input_shape)

    x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)

    x2 = Conv2D(64, (1, 1), activation='relu', padding='same')(input)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x2)

    x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x3 = Conv2D(64, (1, 1), activation='relu', padding='same')(x3)

    x = concatenate([x1, x2, x3], axis=3)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)


def create_resnet50_network(input_shape, weights):
    base_model = ResNet50(input_shape=input_shape, weights=weights)
    return Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)


def create_vgg16true_network(input_shape, weights, dropout1=0.5, dropout2=0.5, \
    l2_reg_strength1=0.0001, l2_reg_strength2=0.0001):
    '''
    Implementation of VGG16 as specified in the original paper by Simonyan
    and Zisserman. Differs from keras' Applications.VGG16 in that it includes
    regularization and dropout on the first two fully connected layers.
    '''

    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape, padding='same',
        activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', \
        kernel_regularizer=regularizers.l2(l2_reg_strength1)))
    model.add(Dropout(dropout1))
    model.add(Dense(4096, activation='relu', \
        kernel_regularizer=regularizers.l2(l2_reg_strength2)))
    model.add(Dropout(dropout2))
    return Model(inputs=model.input, outputs=model.output)

def create(input_shape, network='vgglike', weights=None, builtin_weights=None, \
    dropout1=None, dropout2=None, l2_reg_strength1=None, l2_reg_strength2=None):
    assert network in SUPPORTED_NETWORKS, '%s is an invalid network' % network
    assert weights is None or builtin_weights is None, 'only one type of weights are allowed at once'

    if builtin_weights:
        assert network in SUPPORTED_NETWORKS_WITH_WEIGHTS, \
        '%s does not have weights for %s ' % (network, builtin_weights)

    network_func = globals()['create_%s_network' % network]
    if network == 'vgg16true':
        base_network = network_func(input_shape, builtin_weights, dropout1, \
                        dropout2, l2_reg_strength1, l2_reg_strength2)
    else:
        base_network = network_func(input_shape, builtin_weights)



    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Loading pretrained weights corresponding to the network used
    if weights:
        base_network.load_weights(weights)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    """concatenated = keras.layers.concatenate([processed_a, processed_b])
    out = Dense(1, activation='sigmoid')(concatenated)
    model = Model([input_a, input_b], out)"""

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    return Model([input_a, input_b], distance)



def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compile(model, optimizer, learning_rate, decay, momentum, nesterov, \
    epsilon, loss_func=contrastive_loss):
    assert optimizer in SUPPORTED_OPTIMIZERS, '%s is an invalid optimizer' % optimizer
    if optimizer == 'sgd':
        opt = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    elif optimizer == 'rms':
        opt = RMSprop(lr=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(lr=learning_rate, epsilon=epsilon, decay=decay)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=learning_rate, decay=decay)
    else:
        opt = SUPPORTED_OPTIMIZERS[optimizer]

    model.compile(loss=loss_func, optimizer=opt, metrics=[accuracy])