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
def net_centerloss(bn_size, img_shape, class_num=False, use_net=ResNet50):
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

    # configuer optimizer
    # optimizer = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=True)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        
    if class_num:  # supervision
        # Embedding layer is used to store centers with shape (class_num x dim)
        label = Input(shape=(1,), name='input_label')
        centers = Embedding(class_num, bn_size)(label)
        # calculate loss
        loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([x, centers])
        # update x
        features = Dropout(0.5, name='dropout')(features)
        output = Dense(class_num, activation='softmax', name='dense_output')(features)
        # optimize model and update loss
        model = Model(inputs=[input, label], outputs=[output, loss])
        model.compile(optimizer=optimizer,
                      loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                      loss_weights=[1., 0.001],
                      metrics={'dense_output': 'accuracy'})
    else:  # un-supervision
        output = features
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
