# -*- coding: utf-8 -*-


import os
import re
import random
from shutil import copyfile
import scipy.io
import numpy as np
import cv2
import h5py


# Create DukeMTMC and Market1501 datasets
########################################################
def move_file(source_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for _, _, files in os.walk(source_path, topdown=True):
        for name in files:
            if name.endswith('jpg'):
                src_path = source_path + '/' + name
                dst_path = save_path + '/' + name.split('_')[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)
                if 'val' in dst_path:
                    continue
                else:
                    copyfile(src_path, dst_path + '/' + name)
    return None


def get_dataset_DukeMTMC_and_Market(source_path, save_path):
    # create save folder
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # move images
    move_file(source_path+'/query', save_path+'/query')  # get query data
    move_file(source_path+'/bounding_box_test', save_path+'/gallery')  # get gallery data
    move_file(source_path+'/bounding_box_train', save_path+'/train')  # get train data
    move_file(source_path + '/bounding_box_train', save_path + '/val')  # get train data
    return None

# get_dataset_DukeMTMC_and_Market('../datasets/DukeMTMC', '../datasets/new_DukeMTMC')
# get_dataset_DukeMTMC_and_Market('../datasets/Market1501', '../datasets/new_Market1501')
# print('Successfully create DukeMTMC and Market1501 datasets.')


# Create PRW datasets
########################################################
def split_frames(source_path, save_path, train_path, test_path):
    # get lists of frames for training and test respectively
    train_name = scipy.io.loadmat(train_path)
    test_name = scipy.io.loadmat(test_path)
    lst_train, lst_test = train_name['img_index_train'], test_name['img_index_test']
    # create new paths
    path_frame_train = os.path.join(save_path, 'train_frame')
    if not os.path.isdir(path_frame_train):
        os.mkdir(path_frame_train)
    path_frame_test = os.path.join(save_path, 'test_frame')
    if not os.path.isdir(path_frame_test):
        os.mkdir(path_frame_test)
    # split corresponding frames
    for name in os.listdir(source_path):
        if name.endswith('.jpg'):
            if name.strip('.jpg') in lst_train:
                copyfile(os.path.join(source_path, name), os.path.join(path_frame_train, name))
            elif name.strip('.jpg') in lst_test:
                copyfile(os.path.join(source_path, name), os.path.join(path_frame_test, name))
        print('Now the file {} is moved.'.format(name))


def get_imgs(source_path, info_path, save_path):
    # source_path: store source images
    # info_path: store information of each image from source path
    # save_path: store retrieved images
    # name rule: ID_cxx_xxxxxx (segment could be neglected?)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    count = 0
    for img_name in os.listdir(source_path):
        if img_name.endswith('.jpg'):
            # load the current source image
            img = cv2.imread(os.path.join(source_path, img_name))
            # get information of image
            print('Now the frame is {} ...'.format(img_name))
            try:
                info_mat_path = info_path + '/' + img_name + '.mat'
                info_mat = scipy.io.loadmat(info_mat_path)
            except:
                print('The file is Empty.')
                continue
            try:
                info_mat = info_mat['box_new']
            except:
                info_mat = info_mat['anno_file']
            for i in range(info_mat.shape[0]):
                id, x, y, w, h = info_mat[i, 0], info_mat[i, 1], info_mat[i, 2], info_mat[i, 3], info_mat[i, 4]
                id, x, y, w, h = int(id), int(round(x)), int(round(y)), int(round(w)), int(round(h))
                # retrieve target image for source image according to information (detect person)
                i_img = img[y:y + h, x:x + w, :]
                # get name of image, img_name_new = ID + _ + cx + _ + xxxxxx
                i_name = str(id) + '_' + img_name.split('s')[0] + '_' + img_name.split('_')[1]
                i_path = os.path.join(save_path, str(id))
                # create the folder for the current class
                if not os.path.isdir(i_path):
                    os.mkdir(i_path)
                # save the current image
                cv2.imwrite(os.path.join(i_path, i_name), i_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                count += 1
                print('The {} file {} is saved.'.format(count, i_name))
    return count


def get_query(load_path, save_path):
    count = 0
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for query_name in os.listdir(load_path):
        query_save_path = save_path + '/' + query_name.split('_')[0]
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)
        copyfile(os.path.join(load_path, query_name), os.path.join(query_save_path, query_name))
        print('Query {} is saved in new folder.'.format(query_name))
        count += 1
    return count


def get_val(load_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for train_folder in os.listdir(load_path):
        if train_folder == '.DS_Store':
            continue
        val_name = os.listdir(os.path.join(load_path, train_folder))[0]
        if val_name.endswith('.jpg'):
            if not os.path.isdir(os.path.join(save_path, train_folder)):
                os.mkdir(os.path.join(save_path, train_folder))
            copyfile(os.path.join(load_path, train_folder, val_name), os.path.join(save_path, train_folder, val_name))
            print('Query {} is saved in new folder.'.format(val_name))


# root_load_path = '../datasets/PRW'
# root_save_path = '../datasets/new_PRW'
# if not os.path.isdir(root_save_path):
#     os.mkdir(root_save_path)
#
# split_frames(os.path.join(root_load_path, 'frames'),
#              root_save_path,
#              os.path.join(root_load_path, 'frame_train.mat'),
#              os.path.join(root_load_path, 'frame_test.mat'))
#
# get_imgs(os.path.join(root_save_path, 'train_frame'),
#          os.path.join(root_load_path, 'annotations'),
#          os.path.join(root_save_path, 'train'))
#
# get_imgs(os.path.join(root_save_path, 'test_frame'),
#          os.path.join(root_load_path, 'annotations'),
#          os.path.join(root_save_path, 'gallery'))
#
# get_query(os.path.join(root_load_path,'query_box'), os.path.join(root_save_path,'query'))
#
# get_val(os.path.join(root_save_path, 'train'), os.path.join(root_save_path, 'val'))
#
# print('Successfully create PRW datasets.')


# Create MARS datasets
########################################################

def get_rename_img(src_path, dst_path):
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for folder_name in os.listdir(src_path):
        if folder_name == '.DS_Store':
            continue
        folder_path = os.path.join(dst_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        for img_name in os.listdir(os.path.join(src_path, folder_name)):
            img_name_new = folder_name + '_' + re.findall('C\d', img_name)[0] + '_T' + img_name.split('T')[1]
            save_path = os.path.join(dst_path, folder_name, img_name_new)
            copyfile(os.path.join(src_path, folder_name, img_name), save_path)
            print('Now rename and move the file {} (new name:{}).'.format(img_name, img_name_new))
    return None


def select_query(load_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for test_folder in os.listdir(load_path):
        if (test_folder=='.DS_Store') or (test_folder=='0000') or (test_folder=='00-1'):
            continue
        if not os.path.isdir(os.path.join(save_path, test_folder)):
            os.mkdir(os.path.join(save_path, test_folder))
        lst_img = os.listdir(os.path.join(load_path, test_folder))  # all files in the current folder
        lst_cam = list(set(map(lambda x: re.findall('C\d', x)[0], lst_img)))  # lst_cam = ['C1', 'C2', ...]
        for camera in lst_cam:
            lst_candidate = [i for i in lst_img if camera in i]  # each camera one image
            img_name = random.choice(lst_candidate)
            copyfile(os.path.join(load_path, test_folder, img_name),
                     os.path.join(save_path, test_folder, img_name))
            print('Query {} is saved in new folder.'.format(img_name))
    return None


root_load_path = '../datasets/MARS'
root_save_path = '../datasets/new_MARS'
if not os.path.isdir(root_save_path):
    os.mkdir(root_save_path)

# get_rename_img(os.path.join(root_load_path, 'bbox_train'), os.path.join(root_save_path, 'train'))
# get_rename_img(os.path.join(root_load_path, 'bbox_test'), os.path.join(root_save_path, 'test'))
# get_val(os.path.join(root_save_path, 'train'), os.path.join(root_save_path, 'val'))
# select_query(os.path.join(root_save_path, 'test'), os.path.join(root_save_path, 'query'))
# print('Successfully create MARS datasets.')
