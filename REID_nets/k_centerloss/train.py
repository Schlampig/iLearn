# -*- coding: utf-8 -*-


import os
import random
import numpy as np
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from model import *


# Configure hyper-parameters
###############################################################################
# frequently coordinated hyper-parameters
dataset_name = 'new_Market1501'
use_model = net_centerloss  # used model's function name in model.py
use_net = ResNet50
model_name = 'net_centerloss_ResNet50_new_Market1501.h5'

# assign gpu
use_gpu = True
gpu_id = '1'
if use_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# path
save_path = 'results'
root_path = os.path.join('../datasets/', dataset_name)
train_path = os.path.join(root_path, 'train')
val_path = os.path.join(root_path, 'val')

# image argumentation (mainly for training data)
sample_norm = False  # normalize each sample through its std
shear_range = 0.2  # range of shear mapping
zoom_range = 0.1  # zoom
horizontal_flip = True  # flip along to horizon randomly
data_format = 'channels_last'  # put the dimension of channel in the last place
class_mode = 'categorical'  # task for the model (now is classification)

# model
epoch_num = 10
train_batch_size = 32
val_batch_size = 1
img_shape = (512, 256, 3)
bn_size = 512  # bottleneck size

# Pre-procssing
###############################################################################
# assign gpu
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# get train and validation size
def get_folder_size(path):
    return sum((sum((1 for j in os.listdir(os.path.join(path, i)) if j.endswith('.jpg')))
                for i in os.listdir(path) if i != '.DS_Store'))
train_size = get_folder_size(train_path)
val_size = get_folder_size(val_path)

# pre-process data
train_datagen = ImageDataGenerator(
    samplewise_std_normalization=sample_norm,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    data_format=data_format)

val_datagen = ImageDataGenerator(
    samplewise_std_normalization=sample_norm,
    data_format=data_format)

# build generator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_shape[0], img_shape[1]),
    batch_size=train_batch_size,
    class_mode=class_mode)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_shape[0], img_shape[1]),
    batch_size=val_batch_size,
    class_mode=class_mode)

# get number of classes
class_num = max(train_generator.classes)+1

def get_generator(gen, class_num, batch_size):
    while True:
        X, y = gen.next()  # X with shape (batch_size, dimension), y with shape (batch_size, one_hot)
        y_pre = np.random.randint(0, class_num, batch_size)  # y_pre with shape (batch_size,)
        new_y = np.argwhere(y == 1)[:, 1]  # y_new with shape (batch_size,)
        yield [X, new_y], [y, y_pre]  # [data1, data2, ...], [label1, label2, ...]

# Training and save the model
###############################################################################
# construct model
model = use_model(bn_size=bn_size, img_shape=img_shape, class_num=class_num, use_net=use_net)

# make saving path
if not os.path.isdir(save_path):
    os.mkdir(save_path)
model_save_path = os.path.join(save_path, model_name)

# train and validate model
model.fit_generator(
        get_generator(train_generator, class_num, train_batch_size),
        steps_per_epoch=train_size // train_batch_size,
        epochs=epoch_num,
        validation_data=get_generator(val_generator, class_num, val_batch_size),
        validation_steps=val_size // val_batch_size)

model.save_weights(model_save_path)
print('Successfully train the model: {}.'.format(model_name))

