# -*- coding: UTF-8 -*-

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

'''
DCGAN调优建议：
1. D与G内尽量不用能使特征稀疏的操作，例如G中用ConvTrans(strides=2)替换Conv+UpSampling，用LeakyRelu替换Relu
2. G内建议加Dropout
3. 真假样本不要放在一个mini_batch里训练
4. 建议加BN层
5. 建议使用Adam优化
6. 初始图片特征建议规范化到-1~1间
7. 建议使用soft平滑后的label而不是二值label
来源：https://zhuanlan.zhihu.com/p/28487633
'''


class DCGAN(object):
    def __init__(self, shape_img=(64, 64, 1),
                 shape_noise=(100,),
                 load_path=None,
                 format='.jpg',
                 batch_size=64,
                 epochs=10,
                 shuffle=True,
                 save_path=None,
                 save_interval=5):
        # input: path， the file to store all the images
        #        shape_img, the tuple to store the shape of each image (sometimes is a 3D tensor)
        #        shape_noise, the tuple to store the shape of noise (sometimes is a vector)
        #        format, the postfix of all graphs
        self.shape_img = shape_img
        self.shape_noise = shape_noise
        self.load_path = load_path
        self.format = format
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.save_path = save_path
        self.save_interval = save_interval

    def Adversary(self):
        # 搭建并编译对抗网络模型
        # noise(n,) >> class(1)
        # build the model
        x_in = Input(shape=self.shape_noise)
        x = self.Generator()(x_in)
        x_out = self.Discriminator()(x)
        model = Model(inputs=x_in, outputs=x_out, name='Adversary')
        # compile the model
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='a_model.jpg', show_shapes=True, show_layer_names=True)
        return model

    def Generator(self):
        # 搭建并编译生成器模型
        # noise(n,) >> image(h, w, c)
        # 设置中间层的结构，lst_layer = [layer1, layer2, ..., layerN],
        #                 where layeri = (image_size, kernel_size)
        lst_layer = [(512, 5), (256, 5), (128, 5)]
        x_in = Input(shape=self.shape_noise, name='G0_INPUT')
        x = Dense(8 * 8 * 512, activation='relu', name='G0_ACT')(x_in)
        x = Reshape((8, 8, 512), name='G0_RESHAPE')(x)
        for i, layer in enumerate(lst_layer):
            x = Conv2DTranspose(layer[0], kernel_size=layer[1], strides=2, padding='same', name='G'+str(i+1)+'_CONV')(x)
            # 有说此处用Conv2DTranspose而不是Upsampling，GAN尽量避免稀疏
            # x = Conv2D(layer[0], kernel_size=layer[1], padding='same', name='G' + str(i + 1) + '_CONV')(x)
            # x = UpSampling2D(name='G' + str(i + 1) + '_POOL')(x)
            x = Dropout(0.25, name='G' + str(i + 1) + '_DROP')(x)
            x = Activation('relu', name='G'+str(i+1)+'_ACT')(x)
            x = BatchNormalization(momentum=0.5, name='G'+str(i+1)+'_BN')(x)
        x = Conv2D(self.shape_img[2], kernel_size=5, padding='same', name='G'+str(len(lst_layer)+1)+'_CONV')(x)
        x_out = Activation('tanh', name='G'+str(len(lst_layer)+1)+'_ACT')(x)
        model = Model(inputs=x_in, outputs=x_out, name='Generator')
        plot_model(model, to_file='g_model.jpg', show_shapes=True, show_layer_names=True)
        return model

    def Discriminator(self):
        # 搭建并编译判别器模型
        # image(h, w, c) >> class(1)
        # 设置中间层的结构，lst_layer = [layer1, layer2, ..., layerN],
        #                 where layeri = (image_size, kernel_size)
        lst_layer = [(128, 5), (256, 5), (512, 5)]
        x_in = Input(shape=self.shape_img, name='D0_INPUT')
        x = x_in
        for i, layer in enumerate(lst_layer):
            x = Conv2D(layer[0], kernel_size=layer[1], strides=2, padding='same', name='D'+str(i+1)+'_CONV')(x)
            x = BatchNormalization(momentum=0.5, name='D'+str(i+1)+'_BN')(x)
            x = LeakyReLU(alpha=0.2, name='D'+str(i+1)+'_ACT')(x)
            x = Dropout(0.25, name='D'+str(i+1)+'_DROP')(x)
        x = Flatten(name='D'+str(len(lst_layer)+1)+'_FLATTEN')(x)
        x_out = Dense(1, activation='sigmoid', name='D'+str(len(lst_layer)+1)+'_ACT')(x)
        model = Model(inputs=x_in, outputs=x_out, name='Discriminator')

        optimizer = Adam(lr=0.0002, decay=0.00005)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='d_model.jpg', show_shapes=True, show_layer_names=True)
        return model

    def preprocess_in(self, img):
        # 预处理一个样本，从输入的不规则格式到统一的规则格式
        return transform.resize(img, self.shape_img, mode='reflect')*2.0 - 1

    def preprocess_out(self, img):
        # 预处理一个样本，从数值矩阵生成一幅图
        return 255*(img + 1)/2.0

    def get_batch(self, lst_img, img_tag='d'):
        # 生成一个batch的数据及类别标记
        # input:
        #       lst_img: list of names of the current selected images
        #       img_tag: 'g'-train generator, 'd'-train discriminator
        # output:
        #       X: data, row-batch_size, other dims: image_size
        if img_tag == 'd':
            # 得到一半batch的假样本
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size] + list(self.shape_noise))
            fake_batch = self.Generator().predict(noise)
            # 得到另一半batch的真实样本
            real_batch = []
            for img_name in lst_img:
                img = io.imread(self.load_path + '/' + img_name)
                real_batch.append(self.preprocess_in(img))
            real_batch = np.array(real_batch)
            # 分别返回真假样本及其类别标记,总数是2batch_size：real_batch, real_label, fake_batch, fake_label
            return real_batch, np.ones(real_batch.shape[0]), fake_batch, np.zeros(fake_batch.shape[0])
        elif img_tag == 'g':
            # 得到一个完整batch_size的假样本
            fake_batch = np.random.uniform(-1.0, 1.0, size=[self.batch_size] + list(self.shape_noise))
            # 返回假样本及其类别标记：fake_batch, fake_label
            return fake_batch, np.ones(fake_batch.shape[0])
        else:
            print('Failed to get the image!')
            return None

    def fit(self):
        # 训练整个网络
        # 获取所有样本名称列表
        lst_all = filter(lambda x: x.endswith(self.format), os.listdir(self.load_path+'/'))
        batchs = len(lst_all)/self.batch_size
        # 迭代优化网络
        for i_epoch in range(self.epochs):
            d_acc_sum = 0
            g_acc_sum = 0
            for i_batch in range(batchs):
                # 选择是否打乱读入样本顺序
                if self.shuffle is True:
                    random.shuffle(lst_all)
                # 获取当前batch的样本名列表
                lst_now = lst_all[self.batch_size*i_batch: min(self.batch_size*(i_batch+1),len(lst_all)-1)]
                # 得到真假数据训练判别器
                real_batch, real_label, fake_batch, fake_label = self.get_batch(lst_now, img_tag='d')
                d_loss_real = self.Discriminator().train_on_batch(real_batch, real_label)
                d_loss_fake = self.Discriminator().train_on_batch(fake_batch, fake_label)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # 得到假数据训练生成器
                fake_batch, fake_label = self.get_batch(lst_now, img_tag='g')
                g_loss = self.Adversary().train_on_batch(fake_batch, fake_label)
                # 打印每次训练结果
                print('epoch %d, %d/%d: d_loss-%.4f, d_acc-%.4f, g_loss-%.4f, g_acc-%.4f'
                      % (i_epoch, i_batch, batchs, d_loss[0], d_loss[1]*100, g_loss[0], g_loss[1]*100))
                # 等步长存储图像
                if self.save_path is not None:
                    if i_batch % self.save_interval == 0:
                        self.save_img(i_epoch=i_epoch, i_batch=i_batch)
            print('-'*300)
            print

    def save_img(self, n=3, i_epoch=None, i_batch=None):
        # 绘图并存储
        # n: int, number of fake images generated now
        # i_epoch & i_batch: int
        # 设置图片存储位置
        save_name = self.save_path + '/epoch' + str(i_epoch) + '_batch' + str(i_batch) + self.format
        for i in range(n*n):
            # 得到一个假样本并对其作图
            plt.subplot(n, n, i + 1)
            noise = np.random.uniform(-1.0, 1.0, size=[1] + list(self.shape_noise))
            now_img = self.Generator().predict(noise)
            now_img = self.preprocess_out(now_img)
            now_img = np.reshape(now_img, list(self.shape_img))
            plt.imshow(now_img.squeeze())
        plt.savefig(save_name)
        plt.close()


if __name__ == '__main__':
    clf = DCGAN(shape_img=(64, 64, 1),  # 与G和D的层数有关，不要轻易改
                shape_noise=(100,),
                load_path='./figure',  # 存放所有图片样本的文件夹地址
                format='.jpg',  # 图片后缀（必须统一）
                batch_size=32,
                epochs=200,
                shuffle=True,
                save_path='./figs',  # 存放生成图片的文件夹地址
                save_interval=1)

    clf.fit()
