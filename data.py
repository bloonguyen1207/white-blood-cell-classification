from __future__ import absolute_import, division, print_function, unicode_literals

import time

import numpy as np
import os
import cv2
import random

CATEGORIES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = 168
START_TIME = time.time()

TRAIN_DATA_DIR = "data/dataset2-master/images/TRAIN"
TEST_DATA_DIR = "data/dataset2-master/images/TEST"


def create_data(train=True):
    data = []
    data_dir = TRAIN_DATA_DIR if train else TEST_DATA_DIR

    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), 1)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    random.shuffle(data)
    return data


def separate_features_and_label(data):
    data_x = []
    data_y = []
    for features, label in data:
        data_x.append(features)
        data_y.append(label)

    return reshape_features(data_x), data_y


def reshape_features(features):
    features = np.array(features).reshape([-1, IMG_SIZE, IMG_SIZE, 3])
    features = features / 255.0

    return features
