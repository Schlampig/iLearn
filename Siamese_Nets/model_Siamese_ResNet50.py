# -*- coding: UTF-8 -*-


"""
Siamese network using pre-trained resnet50 as backbone.
"""


import numpy.random as rng
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import l2
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.applications.resnet50 import ResNet50


# Body Model
def get_body(x_input=None, bn_size=128, class_num=False, name=''):
    # bn_size: size of the bottleneck features, interger
    # img_shape: shape of the input image, tuple
    # class_num: number of the classes, interger
    # name = name of output layer of the body
    # x_input = Input() is a tensor
    # x_output is a high-dimensional feature tensor
    assert x_input is not None
    resnet_model = ResNet50(include_top=False,
                            weights='imagenet',
                            classes=class_num)
    resnet_model.name='resnet50'+name
    x = resnet_model(x_input)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dense(bn_size)(x)
    x = BatchNormalization()(x)
    x_output = LeakyReLU(0.1, name='output_layer'+name)(x)
    return x_output


# Siamese Model
def learn_life(bn_size=128, img_shape=(256, 128, 3), class_num=False):
    # training process
    # img_shape: shape of the input image, tuple
    left_head = Input(shape=img_shape)
    left_body = get_body(x_input=left_head, bn_size=bn_size, class_num=class_num, name='_left')
    right_head = Input(shape=img_shape)
    right_body = get_pseudo_body(x_input=right_head, bn_size=bn_size, class_num=class_num, name='_right')
    body = Lambda(lambda x: K.abs(x[0] - x[1]))([left_body, right_body])
    spirit = Dense(1, activation='sigmoid', name='predict_layer')(body)
    life = Model(inputs=[left_head, right_head], outputs=spirit)
    # print model
    life.summary()
    # optimization
    optimizer = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=True)
    life.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return life
