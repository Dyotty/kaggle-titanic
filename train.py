# -*- coding: utf-8 -*-

import csv
import numpy as np
import preprocessing as pp
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import np_utils
from keras.utils import plot_model

import datetime as dt

def read_csv(file_path):
    dst = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            dst.append(row)
    return np.array(dst)


def train_mlp(train_data, label):
    inputs = Input(shape=train_data.shape[1:])
    data = Dense(20, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(inputs)
    data = Dense(20, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(20, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(2, activation="softmax", kernel_initializer="random_uniform", bias_initializer="constant")(data)

    mlp = Model(inputs, data, name="MLP")
    mlp.summary()
    plot_model(mlp, to_file='mlp.png', show_shapes=True)

    mlp.compile(metrics=["accuracy"],
                optimizer="adam",
                loss="categorical_crossentropy")

    mlp.fit(train_data, label,
            batch_size=50,
            epochs=300,
            shuffle="True",
            validation_data=(train_data, label)
            )

    now_time = dt.datetime.now()
    mlp.save("MLP.h5")


# データ読み込み
train_data_path = "dataset/train.csv"
train_data = read_csv(train_data_path)

# 特徴量選定
train_data_selected = pp.preprocessing(train_data)
label = pp.get_label(train_data)
label = np_utils.to_categorical(label)

# 標準化
stdsc = StandardScaler()
train_data_selected = stdsc.fit_transform(train_data_selected)

#
train_mlp(train_data_selected, label)