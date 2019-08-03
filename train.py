# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
from preprocessing import Preprocessing
import models as mdl
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Dense, Input, Conv1D, MaxPool1D, Flatten, BatchNormalization
from keras import regularizers
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import TensorBoard

import datetime as dt

# Try Name
try_name = "MLP_Deep"


def read_csv(file_path):
    dst = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            dst.append(row)
    return np.array(dst)


def train_mlp(train_data, label, weights="", additional_epoch=100):
    mlp = mdl.model_construct_mlp_deep(train_data)

    mlp.compile(metrics=["accuracy"],
                optimizer="adam",
                loss="categorical_crossentropy")

    logdir = "log/" + try_name + dt.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    os.mkdir(logdir)
    callbacks = TensorBoard(log_dir=logdir,
                            histogram_freq=1,
                            write_graph=True,
                            write_grads=True,
                            write_images=True)
    cbks = [callbacks]

    if weights != "":
        mlp = load_model(weights)
        mlp.fit(train_data, label,
                batch_size=50,
                epochs=additional_epoch,
                shuffle="True",
                callbacks=cbks,
                validation_data=(train_data, label)
                )
    else:
        mlp.fit(train_data, label,
                batch_size=50,
                epochs=300,
                shuffle="True",
                validation_data=(train_data, label),
                callbacks=cbks
                # validation_split=0.25
                )

    mlp.save("MLP.h5")

    return mlp


def train_cnn1d(train_data, label):
    cnn1d = mdl.model_construct_cnn1d(train_data)

    cnn1d.compile(metrics=["accuracy"],
                optimizer="adam",
                loss="categorical_crossentropy")

    train_data = np.reshape(train_data, (-1, train_data.shape[1], 1))
    cnn1d.fit(train_data, label,
            batch_size=50,
            epochs=00,
            shuffle="True",
            validation_data=(train_data, label)
            )

    cnn1d.save("CNN1D.h5")

    return cnn1d


def predict_mlp(test_data, model_path):
    model = load_model(model_path)
    result = model.predict(test_data)

    return result


def predict_cnn1d(test_data, model_path):
    test_data = np.reshape(test_data, (-1, test_data.shape[1], 1))
    model = load_model(model_path)
    result = model.predict(test_data)

    return result

# 学習設定
on_off_flg =\
        {
            "Pclass": True,
            "Name": False,
            "Sex": True,
            "Age": True,
            "SibSp": True,
            "Parch": True,
            "Ticket": False,
            "Fare": True,
            "Cabin": False,
            "Embarked": True
        }
model_type = "MLP"


# データ読み込み
train_data_path = "dataset/train.csv"
train_data = read_csv(train_data_path)
# データ前処理クラス初期化
pp = Preprocessing(train_data, mode="train", on_off_list=on_off_flg)

# 特徴量選定
train_data_selected = pp.preprocessing()
label = pp.get_label()
label = np_utils.to_categorical(label)

# 標準化
stdsc = StandardScaler()
train_data_selected = stdsc.fit_transform(train_data_selected)

#
if model_type == "MLP":
    model = train_mlp(train_data_selected, label)
elif model_type == "CNN1D":
    model = train_cnn1d(train_data_selected, label)

#
test_data_path = "dataset/test.csv"
test_data = read_csv(test_data_path)
pp = Preprocessing(test_data, mode="test", on_off_list=on_off_flg)

test_data_selected = pp.preprocessing()
stdsc = StandardScaler()
test_data_selected_std = stdsc.fit_transform(test_data_selected)

if model_type == "MLP":
    result = model.predict(test_data_selected_std)
elif model_type == "CNN1D":
    result = predict_cnn1d(test_data_selected_std, "CNN1D.h5")
result = np.reshape(np.argmax(result, axis=-1), (-1, 1))
PassengerId = np.reshape(np.array(test_data[1:, 0]), (-1, 1))
write_data = np.hstack((PassengerId, result))

with open('result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    for i, _row in enumerate(write_data):
        writer.writerow(_row)