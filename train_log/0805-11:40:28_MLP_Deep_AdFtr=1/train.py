# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import shutil

from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import TensorBoard

from preprocessing import Preprocessing
import models as mdl


def read_csv(file_path):
    dst = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            dst.append(row)
    return np.array(dst)


def train(model, train_data, label, save_dir, weights="", additional_epoch=100):
    # モデル情報を記録
    with open(save_dir + "/model_summary.txt", "w") as fp:
        model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
    plot_model(model, to_file=save_dir + '/model_constructure.png', show_shapes=True)

    # モデルコンパイル
    model.compile(metrics=["accuracy"],
                optimizer="adam",
                loss="categorical_crossentropy")

    # TensorBoard用Callback
    logdir = "log"
    callbacks = TensorBoard(log_dir=logdir,
                            histogram_freq=1,
                            write_graph=True,
                            write_grads=True,
                            write_images=True)
    cbks = [callbacks]

    if weights != "":
        model = load_model(weights)
        fit = model.fit(train_data, label,
                batch_size=50,
                epochs=additional_epoch,
                shuffle="True",
                callbacks=cbks,
                validation_data=(train_data, label)
                )
    else:
        fit = model.fit(train_data, label,
                batch_size=50,
                epochs=300,
                shuffle="True",
                validation_data=(train_data, label),
                callbacks=cbks
                )

    model.save(save_dir + "/" + model.name + ".h5")

    # 学習のログを保存
    plot_history_loss(fit)
    plot_history_acc(fit)
    fig.savefig(save_dir + "/loss-acc.png")
    plt.close()

    return model


# ----------------------------------------------
# Some plots
# ----------------------------------------------
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(20,8))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'])
    axR.plot(fit.history['val_acc'])
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')


# 学習設定
model_lst =\
    [
        mdl.model_construct_mlp_shallow,
        mdl.model_construct_mlp_deep,
        mdl.model_construct_mlp_deep_l2,
        mdl.model_construct_cnn1d
    ]
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
additional_feature =\
    {
        "fare_for_one_person": True
    }
model_type = "MLP"
try_name = "MLP_Deep_AdFtr=1"
model_idx = 1


# データ読み込み
train_data_path = "dataset/train.csv"
train_data = read_csv(train_data_path)

# 特徴量選定と追加特徴量追加
# データ前処理クラス初期化
pp = Preprocessing(train_data, mode="train", on_off_list=on_off_flg, additional_feature_list=additional_feature)
train_data_selected = pp.preprocessing()

# ラベルデータ取得
label = pp.get_label()
label = np_utils.to_categorical(label)

# 学習結果を保存するフォルダを作成
timenow = dt.datetime.now().strftime('%m%d-%H:%M:%S')
save_dir = "train_log/" + timenow + "_" + try_name
os.mkdir(save_dir)

# 学習（学習済モデルとモデル構成情報が指定したディレクトリに保存される）
model = model_lst[model_idx](train_data_selected)
if model_type == "MLP":
    model = train(model, train_data_selected, label, save_dir)
elif model_type == "CNN1D":
    train_data_selected = np.reshape(train_data_selected, (train_data_selected.shape + (1,)))
    model = train(model, train_data_selected, label, save_dir)

# テストデータ読み込み
test_data_path = "dataset/test.csv"
test_data = read_csv(test_data_path)

# テストデータ前処理
pp = Preprocessing(test_data, mode="test", on_off_list=on_off_flg, additional_feature_list=additional_feature)
test_data_selected = pp.preprocessing()

# 予測
if model_type == "MLP":
    result = model.predict(test_data_selected)
elif model_type == "CNN1D":
    train_data_selected = np.reshape(train_data_selected, (train_data_selected.shape + (1,)))
    result = model.predict(test_data_selected)

# 予測結果CSV出力
result = np.reshape(np.argmax(result, axis=-1), (-1, 1))
PassengerId = np.reshape(np.array(test_data[1:, 0]), (-1, 1))
write_data = np.hstack((PassengerId, result))
with open(save_dir + '/result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    for i, _row in enumerate(write_data):
        writer.writerow(_row)

# トライ結果ログ保存
# 設定をtxtファイルに残す
with open(save_dir + '/used_features.txt', 'w') as f:
    # 選定した特徴一覧
    for key in on_off_flg.keys():
        s = key + ": {}".format(on_off_flg[key]) + "\n"
        f.writelines(s)
    # 追加した特徴一覧
    for key in additional_feature.keys():
        s = key + ": {}".format(additional_feature[key]) + "\n"
        f.writelines(s)
# ソースのコピー
shutil.copy("train.py", save_dir + "/train.py")
shutil.copy("preprocessing.py", save_dir + "/preprocessing.py")
shutil.copy("models.py", save_dir + "/models.py")
