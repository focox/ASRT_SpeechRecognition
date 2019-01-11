#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os
import numpy as np
from general_function.file_wav import *
import random
import linecache

class read_data():

    def __init__(self, datalinkpath):
        self.syllable_set = self.read_syllable_dict()
        self.train_list, self.cv_list, self.test_list = self.get_data_list(datalinkpath)
        self.train_info, self.quantity_train = self.integrate_data(self.train_list)
        self.cv_info, self.quantity_cv = self.integrate_data(self.cv_list)
        self.test_info, self.quantity_test = self.integrate_data(self.test_list)
        # self.generate_batch('trian')

    def get_sample(self, data_type, sample_index, audio_length=1600, frequency_length=200):
        """
        get sample data from dataset, and return the following fixed format data.
        :param data_type: train or cv or test.
        :param sample_index: index of sample.
        :param audio_length: the fixed audio length.
        :param frequency_length: the fixed frequency length.
        :return: x, y
        """
        if data_type == 'train':
            data_info = self.train_info
            quantity = self.quantity_train
        elif data_type == 'cv':
            data_info = self.cv_info
            quantity = self.quantity_cv
        elif data_type == 'test':
            data_info = self.test_info
            quantity = self.quantity_test

        # sample_index = random.randint(1, quantity)
        for i in data_info:
            start = data_info[i]['n_start']
            end = start + data_info[i]['quantity']
            if start <= sample_index <= end:
                bias_position = sample_index - start + 1
                wav_path = data_info[i]['wav_path']
                syllable_path = data_info[i]['syllable_path']
                continue
        wav_path_info_ls = linecache.getline(wav_path, bias_position).split(' ')
        wav_name = wav_path_info_ls[0]
        wav_path = wav_path_info_ls[1][: -1]

        syllable_path_info_ls = linecache.getline(syllable_path, bias_position).split(' ')
        syllable_name = syllable_path_info_ls[0]
        syllable_serial = syllable_path_info_ls[1:-1]

        if syllable_name != wav_name:
            raise RuntimeError('the name of pinyin is not same as that of wav at Line%d in file %d and %d' %
                               (bias_position, wav_path, syllable_path))
        y = self.get_syllable_serial(syllable_serial)
        wav, fs = read_wav_data(wav_path)
        # Todo: try the function that is provided.
        wav_feature = GetFrequencyFeature3(wav, fs)
        # print(sample_index, wav_feature.shape)
        x = np.zeros((audio_length, frequency_length), dtype=np.float)
        x[0:wav_feature.shape[0]] = wav_feature
        # Todo: why add the last dimension ?
        # Is it similar the color image signal that consists of 3 channels (RGB) signal.
        x = x.reshape(x.shape[0], x.shape[1], 1)
        return x, y

    def generate_batch(self, data_type, batch_size=32, shuffle=True):
        """
        generate data of batch_size, when all data has been got, epoch = True
        :param data_type: train or cv or test
        :param batch_size: batch size
        :param shuffle: if shuffle is True, generate data randomly.
        :return: [x, y], epoch
        """
        if data_type == 'train':
            quantity = self.quantity_train
        elif data_type == 'cv':
            quantity = self.quantity_cv
        elif data_type == 'test':
            quantity = self.quantity_test

        # get one sample to obtain the shapes of x and y. And sample_index could be any valid index.
        x_refer, y_refer = self.get_sample(data_type=data_type, sample_index=1)
        x_shape = x_refer.shape
        y_shape = y_refer.shape
        x = np.zeros((batch_size, x_shape[0], x_shape[1], x_shape[2]), dtype=np.float)
        y = np.zeros((batch_size, y_shape[0], y_shape[1]), dtype=np.float)

        sample_index = np.arange(quantity)
        if shuffle:
            np.random.shuffle(sample_index)
        end = 0
        while True:
            start = end
            end = start + batch_size
            if end >= quantity-1:
                samples = sample_index[start:]
                rest_n_sample = batch_size - len(samples)
                samples = np.append(samples, sample_index[:rest_n_sample])
                end = rest_n_sample
                epoch = True
            else:
                samples = sample_index[start:end]
                epoch = False
            for idx, i in enumerate(samples):
                x[idx], y[idx] = self.get_sample(data_type, sample_index=i)
            # print([x, y], epoch)
            yield [x, y]

    def get_syllable_serial(self, chars, syllable_label_length=64):
        """
        transform all chars into the index of syllable in the dict, and then generate the numpy.array
        :param chars: the all pinyin
        :param syllable_label_length: the fixed length of syllable lable.
        :return: numpy.array,
        """
        serials = []
        length = 0
        for n, char in enumerate(chars):
            serials.append(self.syllable2number(char))
            length = n + 1
        bias = length - syllable_label_length  # 64 is the max length of syllable_serial.
        length = len(serials)
        if bias > 0:
            print("the length of serial exceeds 64")
        else:
            for i in range(abs(bias)):
                serials.append(0)
        return np.array(serials, dtype=np.float)

    def syllable2number(self, char):
        """
        transform syllable into number that is numbered by the index of char in the pinyin list.
        :param char: syllable
        :return: the index of char in the pinyin list.
        """
        return self.syllable_set.index(char)

    def read_syllable_dict(self):
        """
        read the pinyin file from pinyin dict
        :return: pinyin list that only consists of pinyin including tone.
        """
        syllable_dict_path = os.path.join('.', 'dict.txt')  # this operation can be compatible on any system platform.
        with open(syllable_dict_path, mode='r') as f:
            syllable_list = f.readlines()
        syllable_set = []
        for i in syllable_list:
            if i != '':
                char = i.split('\t')
                syllable_set.append(char[0])
        syllable_set.append('_')  # append the character '_' at the end of original list.
        return syllable_set

    @staticmethod
    def integrate_data(data_list):
        """
        integrate data of data_type
        :param data_list: e.g. [['datalist/st-cmds/train.syllable.txt', 'datalist/st-cmds/train.wav.txt'],...]
        :param data_type: train, cv or test
        :return: {1:{'wav_path': wav_link, 'syllable_path': syllable_link, 'quantity': n_wav_link, 'n_start': start}, 2:}
        """
        data_info = {}
        start = 1
        for num, link in enumerate(data_list):
            syllable_link, wav_link = link[0], link[1]
            with open(syllable_link, mode='r') as f:
                syllable_path = f.readlines()
            with open(wav_link, mode='r') as f:
                wav_path = f.readlines()
            n_syllable_link = len(syllable_path)
            n_wav_link = len(wav_path)
            if n_syllable_link != n_wav_link:
                raise RuntimeWarning("The number of file %s doesn't equal to that of file %s " % (syllable_link, wav_link))
            else:
                data_info[num] = {'wav_path': wav_link, 'syllable_path': syllable_link, 'quantity': n_wav_link, 'n_start': start}
                start += n_wav_link
        quantity_data_type = start - 1
        return data_info, quantity_data_type

    @staticmethod
    def get_data_list(datalinkpath):
        """
        get all paths of links that point all real paths of dataset, and return the train_link, cv_link, test_link.
        :param datalinkpath: the path of link of dataset.
        :return:[[path of train_syllable, path of train_wav], []]
                [[path of cv_syllable, path of cv_wav], []]
                [[path of test_syllable, path of test_wav], []]
        """
        train_list, cv_list, test_list = [], [], []
        dirs = os.listdir(datalinkpath)
        for dir in dirs:
            dirpath = os.path.join(datalinkpath, dir)
            file_list = [i[2] for i in os.walk(dirpath)][0]
            sub_train_list, sub_cv_list, sub_test_list = [], [], []
            for file in file_list:
                if 'train' in file:
                    sub_train_list.append(os.path.join(datalinkpath, dir, file))
                elif 'dev' in file or 'cv' in file:
                    sub_cv_list.append(os.path.join(datalinkpath, dir, file))
                elif 'test' in file:
                    sub_test_list.append(os.path.join(datalinkpath, dir, file))
            train_list.append(sub_train_list)
            cv_list.append(sub_cv_list)
            test_list.append(sub_test_list)
        return train_list, cv_list, test_list

    @staticmethod
    def dataset_consistency_check(datalinkpath):
        pass

if __name__ == '__main__':
    test = read_data('dataset')