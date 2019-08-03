# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessing:

    def __init__(self, data, mode, on_off_list):
        self.data = data
        self.mode = mode
        self.on_off_flg = on_off_list

        # trainとtestでデータ形式が変わる（ラベルがない分）ので調整
        if mode == "train":
            self.idx_feature_start = 2
        elif mode == "test":
            self.idx_feature_start = 1

        # 特徴名のリスト（後で使う。ここで宣言しないと面倒）
        self.feature_name_lst = data[0, self.idx_feature_start:]

        # 特徴名ごとの列インデックスdictionary
        self.feature_name_dct = {}
        for i, name in enumerate(self.feature_name_lst):
            self.feature_name_dct[name] = i

        # 特徴データが書かれている列以降のみにする
        self.feature_data = data[1:, self.idx_feature_start:]

    # 新規特徴量
    # 1人あたりの料金
    def fare_for_one_person(self):
        # 必要なデータを前処理
        sibsp_data = self.preprocessing_with_key("SibSp")
        parch_data = self.preprocessing_with_key("Parch")
        fare_data = self.preprocessing_with_key("Fare")
        # 一人あたりの料金を計算
        fare_for_one_person = fare_data / (sibsp_data + parch_data + 1)
        # hstack結合の為に二次元にしておく
        fare_for_one_person = np.reshape(fare_for_one_person, (-1, 1))

        return fare_for_one_person

    def get_label(self):
        if self.mode == "train":
            return np.array(self.data[1:, 1], np.float64)

    def preprocessing(self):
        # 使用する特徴を切り替える為に特徴名のdictionaryを作成
        feature_name_dct = {}
        for i, name in enumerate(self.feature_name_lst):
            feature_name_dct[name] = i

        # 使用する特徴を切り替える為に、残った特徴名listを作成
        remain_name_lst = []
        for i, name in enumerate(self.feature_name_lst):
            if self.on_off_flg[name]:
                remain_name_lst.append(name)

        # 使用する特徴のみ残して前処理を行う
        # 使用する特徴名でループ、前処理を行い結果をappendしていく
        selected_feature_data_lst = []
        for name in remain_name_lst:
            one_feature_data = self.preprocessing_with_key(name)
            selected_feature_data_lst.append(one_feature_data)
        selected_feature_data = np.array(selected_feature_data_lst)     # ndarrayに変換
        selected_feature_data = selected_feature_data.T                 # 転置して(891,*)のshapeにする

        # np.float64に変換
        dst = np.asarray(selected_feature_data, np.float64)

        # 新規特徴量を追加
        fare_for_one_person = self.fare_for_one_person()
        dst = np.hstack((dst, fare_for_one_person))

        # 標準化（スケール統一）
        stdsc = StandardScaler()
        dst = stdsc.fit_transform(dst)

        return dst

    def preprocessing_with_key(self, key):
        func_dct =\
            {
                "Pclass": self.prepro_Pclass,
                "Name": self.prepro_Name,
                "Sex": self.prepro_Sex,
                "Age": self.prepro_Age,
                "SibSp": self.prepro_SibSp,
                "Parch": self.prepro_Parch,
                "Ticket": self.prepro_Ticket,
                "Fare": self.prepro_Fare,
                "Cabin": self.prepro_Ticket,
                "Embarked": self.prepro_Embarked
            }
        return func_dct[key](self.feature_data[:, self.feature_name_dct[key]])

    def prepro_Pclass(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = -1
            else:
                dst[i] = val
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_Name(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = -1
            else:
                dst[i] = val
        return dst

    def prepro_Sex(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "male":
                dst[i] = 0
            elif val == "female":
                dst[i] = 1
            else:
                dst[i] = -1
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_Age(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = 24
            else:
                dst[i] = val
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_SibSp(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = 0
            else:
                dst[i] = val
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_Parch(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = 0
            else:
                dst[i] = val
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_Fare(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = 0
            else:
                dst[i] = val
        dst = np.asarray(dst, np.float64)
        return dst

    def prepro_Ticket(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = -1
            else:
                dst[i] = val
        return dst

    def prepro_Cabin(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "":
                dst[i] = -1
            else:
                dst[i] = val
        return dst

    def prepro_Embarked(self, data):
        dst = np.zeros(data.shape)
        for i, val in enumerate(data):
            if val == "Q":
                dst[i] = 0
            if val == "C":
                dst[i] = 0.5
            if val == "S":
                dst[i] = 1
        dst = np.asarray(dst, np.float64)
        return dst
