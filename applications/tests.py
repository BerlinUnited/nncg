from nncg.nncg import NNCG
from applications.daimler.loader import random_imdb
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, MaxPooling2D, Convolution2D, Dropout, Dense
from tensorflow.keras.models import Sequential

tf.compat.v1.disable_eager_execution()


def print_success(name):
    """
    Prints that a test has passed.
    :param name: Name of the test.
    :return: None.
    """
    print('''
    
######################################################################
            {} passed
######################################################################    
    
'''.format(name))


def no_dense_test():
    """
    Tests an example CNN with no dense layer.
    :return: None
    """
    num_imgs = 10
    nncg = NNCG()
    no_dense = Sequential()
    no_dense.add(Convolution2D(4, (3, 3), input_shape=(36, 18, 1),
                               activation='relu', padding='same'))
    no_dense.add(MaxPooling2D(pool_size=(2, 2)))
    no_dense.add(Convolution2D(8, (3, 3), padding='same', activation='relu'))
    no_dense.add(MaxPooling2D(pool_size=(2, 2)))
    no_dense.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))  # Could be softmax
    no_dense.add(MaxPooling2D(pool_size=(4, 2)))
    no_dense.add(Dropout(0.4))
    no_dense.add(Convolution2D(2, (2, 2), activation='softmax'))
    no_dense.add(Flatten())
    images = random_imdb(num_imgs, no_dense.input.shape[1:].as_list())
    nncg.keras_compile(images, no_dense, 'no_dense.c')
    print_success('no_dense')


def dense_test():
    """
    Tests an example CNN with a Dense layer and valid padding.
    :return: None.
    """
    num_imgs = 10
    nncg = NNCG()
    dense_model = Sequential()
    dense_model.add(Convolution2D(4, (3, 3), input_shape=(70, 50, 1),
                                  activation='relu', padding='same'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Convolution2D(8, (3, 3), padding='valid', activation='relu'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Convolution2D(16, (3, 3), padding='valid', activation='relu'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Dropout(0.4))
    dense_model.add(Flatten())
    dense_model.add(Dense(2, activation='softmax'))
    images = random_imdb(num_imgs, dense_model.input.shape[1:].as_list())
    nncg.keras_compile(images, dense_model, 'dense_model.c')
    print_success('dense_model')


def strides_test():
    """
    Tests an example CNN with additional unusual strides.
    :return: None.
    """
    num_imgs = 10
    nncg = NNCG()
    strides_model = Sequential()
    strides_model.add(Convolution2D(4, (3, 3), input_shape=(101, 101, 1),
                                    activation='relu', padding='same', strides=(3, 3)))
    strides_model.add(MaxPooling2D(pool_size=(2, 2)))
    strides_model.add(Convolution2D(8, (3, 3), padding='valid', activation='relu', strides=(2, 3)))
    strides_model.add(Convolution2D(16, (3, 3), padding='valid', activation='relu'))
    strides_model.add(Flatten())
    strides_model.add(Dense(2, activation='softmax'))
    images = random_imdb(num_imgs, strides_model.input.shape[1:].as_list())
    nncg.keras_compile(images, strides_model, 'strides.c')
    print_success('strides')


def vgg16_test():
    """
    Tests a full VGG16.
    :return: None.
    """
    num_imgs = 1
    nncg = NNCG()
    vgg16_m = VGG16(weights=None)
    images = random_imdb(num_imgs, vgg16_m.input.shape[1:].as_list())
    nncg.keras_compile(images, vgg16_m, 'vgg16.c', weights_method='stdio')
    print_success('VGG16')


def vgg19_test():
    """
    Tests a full VGG19.
    :return: None.
    """
    num_imgs = 1
    nncg = NNCG()
    vgg19_m = VGG19(weights=None)
    images = random_imdb(num_imgs, vgg19_m.input.shape[1:].as_list())
    nncg.keras_compile(images, vgg19_m, 'vgg19.c', weights_method='stdio')
    print_success('VGG19')


if __name__ == '__main__':
    # All tests do not need an image database so we just call them.
    no_dense_test()
    dense_test()
    strides_test()
    vgg16_test()
    vgg19_test()
