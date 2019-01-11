#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os
import time
from typing import List, Any, Generator

from numpy.core.multiarray import ndarray

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import keras as kr
import numpy as np
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization  # , Flatten
from keras.layers import Lambda, TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from keras.callbacks import ModelCheckpoint

from read_data import read_data

ModelName = '26'

# NUM_GPU = 2


class ModelSpeech():  # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        '''
        self.datapath = datapath
        MS_OUTPUT_SIZE = 1422
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.CreateModel()

    def CreateModel(self):
        """
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        :return:
        """


        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        # layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
        layer_h3 = Dropout(0.05)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h16 = Reshape((200, 3200))(layer_h15)  # Reshape层
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        # layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        # layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        model.summary()

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        # opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        # print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred[:, :, :]
        # y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def TrainModel(self, modelpath, epochs=2, save_step=1000, batch_size=32):

        checkpoint = ModelCheckpoint(modelpath, monitor='val_loss', save_weights_only=True, verbose=1,
                                     save_best_only=True, period=1)
        data = read_data(self.datapath)
        train_x_y = data.generate_batch('train', batch_size=batch_size)
        cv_x_y = data.generate_batch('cv', batch_size=batch_size)
        self._model.fit_generator(train_x_y, steps_per_epoch=5, epochs=1, verbose=1, callbacks=[checkpoint],
                                  class_weight=None, max_queue_size=10,
                                  workers=1, use_multiprocessing=True, shuffle=True, initial_epoch=0)

    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''

        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)

        in_len[0] = input_len

        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.base_model.predict(x=x_in)

        # print('base_pred:\n', base_pred)

        # y_p = base_pred
        # for j in range(200):
        #	mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
        #	print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
        #	print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
        #	count=0
        #	for i in range(y_p[0][j].shape[0]):
        #		if(y_p[0][j][i] < mean):
        #			count += 1
        #	print('count:',count)

        base_pred = base_pred[:, :, :]
        # base_pred =base_pred[:, 2:, :]

        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        # print('r', r)

        r1 = K.get_value(r[0][0])
        # print('r1', r1)

        # r2 = K.get_value(r[1])
        # print(r2)

        r1 = r1[0]

        return r1
        pass

    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        不过这里现在还有bug
        '''

        # data = self.data
        # data = DataSpeech('E:\\语音数据集')
        # data.LoadDataList('dev')
        # 获取输入特征
        # data_input = GetMfccFeature(wavsignal, fs)
        # t0=time.time()
        data_input = GetFrequencyFeature3(wavsignal, fs)
        # t1=time.time()
        # print('time cost:',t1-t0)

        input_length = len(data_input)
        input_length = input_length // 8

        data_input = np.array(data_input, dtype=np.float)
        # print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        # t2=time.time()
        r1 = self.Predict(data_input, input_length)
        # t3=time.time()
        # print('time cost:',t3-t2)
        list_symbol_dic = GetSymbolList(self.datapath)  # 获取拼音列表

        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str
        pass

    def RecognizeSpeech_FromFile(self, filename):
        '''
        最终做语音识别用的函数，识别指定文件名的语音
        '''

        wavsignal, fs = read_wav_data(filename)

        r = self.RecognizeSpeech(wavsignal, fs)

        return r

        pass

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model


if __name__ == '__main__':

    ms = ModelSpeech('dataset')

    ms.TrainModel('/m6/test.model')

# t1=time.time()
# ms.TestModel(datapath, str_dataset='train', data_count = 128, out_report = True)
# ms.TestModel(datapath, str_dataset='dev', data_count = 128, out_report = True)
# ms.TestModel(datapath, str_dataset='test', data_count = 128, out_report = True)
# t2=time.time()
# print('Test Model Time Cost:',t2-t1,'s')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0053.wav')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00020I0087.wav')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\train\\A11\\A11_167.WAV')
# r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
# print('*[提示] 语音识别结果：\n',r)
