# -*- coding: UTF-8 -*-


"""
*  Only basic model is given here.
** The code is inspired by the following methods:
   [1] https://github.com/ascourge21/Siamese
   [2] https://github.com/abartnof/siamese_exemplars/blob/master/siamese_omniglot_final.ipynb
"""


import numpy.random as rng
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import l2
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.applications.resnet50 import ResNet50


# Weights initialization
def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


# Backbone
def get_body(x_input=None, img_shape=(256, 128, 3), name=''):
    # img_shape: shape of the input image, tuple
    # name = name of output layer of the body
    # x_input = Input() is a tensor
    # x_output is a high-dimensional feature tensor
    assert x_input is not None
    x = Conv2D(64, (10, 10),
               activation='relu', input_shape=img_shape,
               kernel_initializer=W_init,
               kernel_regularizer=l2(2e-4))(x_input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7),
               activation='relu',
               kernel_regularizer=l2(2e-4),
               kernel_initializer=W_init,
               bias_initializer=b_init)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4),
               activation='relu',
               kernel_initializer=W_init,
               kernel_regularizer=l2(2e-4),
               bias_initializer=b_init)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4),
               activation='relu',
               kernel_initializer=W_init,
               kernel_regularizer=l2(2e-4),
               bias_initializer=b_init)(x)
    x_fea = Flatten()(x)
    x_output = Dense(512,
                     activation="sigmoid",
                     kernel_regularizer=l2(1e-3),
                     kernel_initializer=W_init,
                     bias_initializer=b_init,
                     name=name+'output_layer')(x_fea)
    return x_output


# Siamese Model
def get_life(img_shape):
    # training process
    # img_shape: shape of the input image, tuple
    left_head = Input(shape=img_shape)
    left_body = get_pseudo_body(x_input=left_head, img_shape=img_shape, name='left_')
    right_head = Input(shape=img_shape)
    right_body = get_body(x_input=right_head, img_shape=img_shape, name='right_')
    body = Lambda(lambda x: K.abs(x[0] - x[1]))([left_body, right_body])
    spirit = Dense(1, activation='sigmoid', bias_initializer=b_init)(body)
    life = Model(inputs=[left_head, right_head], outputs=spirit)
    # optimization
    optimizer = Adam(lr=0.00006)
    life.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return life


def learn_live(img_shape):
    # test process
    # img_shape: shape of the input image, tuple
    head = Input(shape=img_shape, name='input_test')
    body = get_pseudo_body(x_input=head, img_shape=img_shape, name='')
    live = Model(inputs=head, outputs=body)
    return live
