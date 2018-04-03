# -*- coding: UTF-8 -*-


import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K
from keras.engine.topology import Layer
from keras.utils import *
# default network
from keras.applications.resnet50 import ResNet50
# VGGX needs special image preprocessings
from keras.applications.vgg16 import VGG16
# Xception might not be suitable for the reid problem
from keras.applications.xception import Xception
# following nets are only provided by 2.1.3 or higher versions
# from keras.applications.densenet import DenseNet121
# from keras.applications.inception_resnet_v2 import InceptionResNetV2


# Define layer
class SphereLoss(Layer):
    def __init__(self, output_dim, m=2, **kwargs):
        self.class_num = output_dim
        self.m = m
        super(SphereLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.class_num), initializer='uniform')
        super(SphereLoss, self).build(input_shape)

    def call(self, x, mask=None):
        # S = each row contains |x|cos(x, W), with shape (None, class_num)
        # S_m: each row contains |x|cos(m, x, W), with shape (None, class_num)
        # S_new = combine(S[except col_i], S_m[col_i]), with shape (None, class_num)
        # S_sum_now = sum(S_new, axis=1), with shape (None, 1)
        # S_soft_now = S_m / S_sum_now, with shape (None, 1)
        # output = combine class_num S_soft_now, with shape (None, class_num)

        turb = 1e-7
        X = x  # X with shape (None, dim), y with shape (None, class_num)

        # calculate S
        W_norm = K.l2_normalize(self.W, axis=0)  # W and W_norm with shape(dim, class_num)
        S = K.dot(X, W_norm)  # S with shape(None, class_num)

        # calculate theta
        X_norm = K.sqrt(K.sum(K.square(X), axis=1))  # turn x_i to |x_i|, hence X becomes X_norm with shape (None, 1)
        X_norm = K.reshape(X_norm, (-1, 1))
        X_norm = K.tile(X_norm, (1, self.class_num))  # |x|, with shape (None, 1)
        X_norm_turb = 1 / (X_norm + turb)
        theta = tf.multiply(X_norm_turb, S)  # theta here denotes cosθ with shape (None, class_num)

        # calculate cos(mθ), theta_m with shape(None, class_num)
        if self.m == 1:
            # cosθ
            theta_m = theta
        elif self.m == 2:
            # 2 * cosθ^2 - 1
            theta_m = 2 * K.square(theta) - 1
        elif self.m == 3:
            # 4 * cosθ^2 - 3 * cosθ
            theta_m = 4 * K.square(theta) - 3 * theta
        elif self.m == 4:
            # 8 * cosθ^4 - 8 * cosθ^2 - 1
            theta_m = 8 * K.pow(theta, 4) - 8 * K.square(theta) - 1
        else:
            return None

        # calculate S_m
        S_m = tf.multiply(X_norm, theta_m)  # S_m = |x|cos(m, x, W), with shape (None, class_num)

        # # calculate output with shape (None, class_num)
        for i_class in range(self.class_num):
            index_mask = [j for j in range(self.class_num) if j != i_class]
            S_m_now = K.reshape(S_m[:, i_class], (-1, 1))
            S_now = tf.gather(S_m, index_mask, axis=1)
            S_new = K.concatenate([S_m_now, S_now], axis=1)
            S_sum_now = K.reshape(K.sum(S_new, axis=1), (-1, 1))  # S_sum_now with shape (None, 1)
            S_soft_now = tf.div(S_m_now, S_sum_now)
            if i_class == 0:
                output = S_soft_now
            else:
                output = K.concatenate([output, S_soft_now], axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.class_num

    
# Model
def net_sphereloss(bn_size, img_shape, class_num=False, use_net=ResNet50):
    # img_shape: shape of the input image
    # bn_size: size of the bottleneck features
    # class_num: number of the classes
    # use_net: name of the function, e.g., VGG16, ResNet50, ...

    # 正常网络
    input = Input(shape=img_shape, name='input')
    x = use_net(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=class_num)(input)
    x = GlobalAveragePooling2D(data_format='channels_last', name='global_avg_pool2d')(x)
    x = Dense(bn_size, name='dense_1')(x)
    x = BatchNormalization(name='batch_norm')(x)
    features = LeakyReLU(0.1, name='leaky_relu')(x)

    # configure optimizer
    # optimizer = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=True)
    optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
    
    if class_num:  # supervision
        # update x
        output = SphereLoss(output_dim=class_num, m=4)(features)
        # optimize model and update loss
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:  # un-supervision
        output = features
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

