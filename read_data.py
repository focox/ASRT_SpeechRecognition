#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os
import numpy as np
from general_function.file_wav import *
import random

class DataSpeech():

    def __init__(self, datalinkpath):
        self.train_list, self.cv_list, self.test_list = self.get_data_list(datalinkpath)
        print(self.train_list, self.cv_list, self.test_list)

    @staticmethod
    def get_data_list(datalinkpath):
        """
        get all paths of links that point all real paths of dataset, and return the train_link, cv_link, test_link.
        :param datalinkpath: the path of link of dataset
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
    test = DataSpeech('datalist')