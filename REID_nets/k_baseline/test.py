# -*- coding: utf-8 -*-


import os
import re
import scipy.io
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from model import *


# Configure hyper-parameters
###############################################################################
# frequently coordinated hyper-parameters
dataset_name = 'new_Market1501'
use_model = baseline  # used model's function name in model.py
use_net = ResNet50
model_name = 'baseline_ResNet50_new_Market1501.h5'

# assign gpu
use_gpu = True
gpu_id = '1'
if use_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# path
load_path = 'results'
save_path = os.path.join(load_path,'features_' + model_name.strip('.h5'))
weights_path = os.path.join(load_path, model_name)
root_path = os.path.join('../datasets/', dataset_name)
gallery_path = os.path.join(root_path, 'gallery')
query_path = os.path.join(root_path, 'query')

# image argumentation (mainly for training data)
data_format = 'channels_last'

# model
layer_name = 'leaky_relu'
epoch_num = 10
gallery_batch_size = 32
query_batch_size = 2
img_shape = (512, 256, 3)
bn_size = 512


# Pre-procssing
###############################################################################
# get gallery and query size
def get_folder_size(path):
    return sum((sum((1 for j in os.listdir(os.path.join(path, i)) if j.endswith('.jpg')))
                for i in os.listdir(path) if i != '.DS_Store'))
gallery_size = get_folder_size(gallery_path)
query_size = get_folder_size(query_path)

# pre-process data
gallery_datagen = ImageDataGenerator(data_format=data_format)
query_datagen = ImageDataGenerator(data_format=data_format)

# build generator
gallery_generator = gallery_datagen.flow_from_directory(
    gallery_path,
    target_size=(img_shape[0], img_shape[1]),
    batch_size=gallery_batch_size,
    shuffle=False)

query_generator = query_datagen.flow_from_directory(
    query_path,
    target_size=(img_shape[0], img_shape[1]),
    batch_size=query_batch_size,
    shuffle=False)

# get number of classes
class_num = max(gallery_generator.classes)+1


# Using model to generate features
###############################################################################
# load model (and its weights)
model = use_model(bn_size, img_shape, use_net=use_net)
model.load_weights(weights_path, by_name=True)
model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# run model
gallery_features = model.predict_generator(gallery_generator, verbose=1)
query_features = model.predict_generator(query_generator, verbose=1)

# get corresponding labels and cameras
# sample name form: class_camera_code.jpg, e.g., 0002_c3_273je28.jpg
gallery_feature, gallery_name = gallery_features, gallery_generator.filenames
query_feature, query_name = query_features, query_generator.filenames
def get_label_and_cam(lst_name):
    lst_label, lst_cam = [], []
    for name_now in lst_name:
        label_now, name_now = int(name_now.split('/')[0]), name_now.split('/')[1]
        cam_now = int(re.findall('[C, c](\d+)', name_now)[0])
        if label_now and cam_now:
            # print(name_now, ':', label_now, cam_now)
            lst_label.append(label_now)
            lst_cam.append(cam_now)
    return lst_label, lst_cam
gallery_label, gallery_cam = get_label_and_cam(gallery_name)
query_label, query_cam = get_label_and_cam(query_name)
assert gallery_label.shape[0] == gallery_feature.shape[0]
assert query_label.shape[0] == query_feature.shape[0]

# save the generated data
test_data = {'gallery_feature': gallery_feature,
             'gallery_label': gallery_label,
             'gallery_cam': gallery_cam,
             'query_feature': query_feature,
             'query_label': query_label,
             'query_cam': query_cam}
scipy.io.savemat(save_path, test_data)
print('Successfully get features of the gallery and query data.')
