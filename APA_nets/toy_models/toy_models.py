# -*- coding: UTF-8 -*-


from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import *
import keras.backend as K
from keras.utils.vis_utils import plot_model
# default network
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


# Models
##############################################################################
def DAN(img_shape=(48, 48, 3), class_num=3):
    # img_shape: shape of the input image, tuple (h, w, c)
    # class_num: number of classes, integer
    pre_model = VGG16(input_shape=img_shape, include_top=False, weights=None, classes=class_num)
    layer = pre_model.get_layer('block5_pool').output

    x_avg_in = GlobalAveragePooling2D(data_format='channels_last')(layer)
    x_avg_in = Lambda(lambda p: K.l2_normalize(p))(x_avg_in)
    x_max_in = GlobalMaxPooling2D(data_format='channels_last')(layer)
    x_max_in = Lambda(lambda p: K.l2_normalize(p))(x_max_in)

    x = Concatenate()([x_max_in, x_avg_in])
    x_output = Dense(class_num, activation='sigmoid')(x)

    model = Model(inputs=pre_model.input, outputs=x_output)
    plot_model(model, to_file='DAN.jpg', show_shapes=True, show_layer_names=True)
    return model


def DAN_plus(img_shape=(48, 48, 3), class_num=3):
    # img_shape: shape of the input image, tuple (h, w, c)
    # class_num: number of classes, integer
    pre_model = VGG16(input_shape=img_shape, include_top=False, weights=None, classes=class_num)
    layer_out = pre_model.get_layer('block5_conv2').output
    layer_in = pre_model.get_layer('block5_pool').output

    x_avg_in = GlobalAveragePooling2D(data_format='channels_last')(layer_in)
    x_avg_in = Lambda(lambda p: K.l2_normalize(p))(x_avg_in)
    x_max_in = GlobalMaxPooling2D(data_format='channels_last')(layer_in)
    x_max_in = Lambda(lambda p: K.l2_normalize(p))(x_max_in)
    x_avg_out = GlobalAveragePooling2D(data_format='channels_last')(layer_out)
    x_avg_out = Lambda(lambda p: K.l2_normalize(p))(x_avg_out)
    x_max_out = GlobalMaxPooling2D(data_format='channels_last')(layer_out)
    x_max_out = Lambda(lambda p: K.l2_normalize(p))(x_max_out)

    x = Concatenate()([x_max_out, x_max_in, x_avg_in, x_avg_out])
    x_output = Dense(class_num, activation='sigmoid')(x)

    model = Model(inputs=pre_model.input, outputs=x_output)
    plot_model(model, to_file='DAN_plus.jpg', show_shapes=True, show_layer_names=True)
    return model


# Tests
##############################################################################
class_num = 5
train_size = 500
batch_size = 32
img_shape = (64, 64, 3)
epoch_num = 5

X = np.random.random((train_size, img_shape[0], img_shape[1], img_shape[2]))
y = np.random.randint(0, class_num, train_size)
y = to_categorical(y)

model = DAN_plus(img_shape=img_shape, class_num=class_num)
optimizer = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=epoch_num)
