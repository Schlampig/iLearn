# -*- coding: UTF-8 -*-


from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
# default network
from keras.applications.resnet50 import ResNet50
# VGGX needs special image preprocessings
from keras.applications.vgg16 import VGG16
# Xception might not be suitable for the reid problem
from keras.applications.xception import Xception
# following nets are only provided by 2.1.3 or higher versions
# from keras.applications.densenet import DenseNet121
# from keras.applications.inception_resnet_v2 import InceptionResNetV2


# Model
def baseline(bn_size, img_shape, class_num=False, use_net=ResNet50):
    # img_shape: shape of the input image, tuple
    # bn_size: size of the bottleneck features, interger
    # class_num: number of the classes, interger
    # use_net: function
    input = Input(shape=img_shape, name='input')
    # load pre-trained network
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
    if class_num:
        features = Dropout(0.5, name='dropout')(features)
        output = Dense(class_num, activation='softmax', name='dense_output')(features)
    else:
        output = features
    model = Model(input, output)
    # plot_model(model, to_file='preid_net.jpg', show_shapes=True, show_layer_names=True)

    optimizer = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
