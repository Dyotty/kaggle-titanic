# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocessing(data):
    return data


def preprocessing_with_key(key, data):
    func_dct =\
        {
            "Pclass": prepro_Pclass,
            "Name": prepro_Name,
            "Sex": prepro_Sex,
            "Age": prepro_Age,
            "SibSp": prepro_SibSp,
            "Parch": prepro_Parch,
            "Ticket": prepro_Ticket,
            "Cabin": prepro_Ticket,
            "Embarked": prepro_Embarked
        }
    return func_dct[key](data)


def prepro_Pclass(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Name(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Sex(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "male":
            dst[i] = 0
        elif val == "female":
            dst[i] = 1
        else:
            dst[i] = -1
    return dst


def prepro_Age(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_SibSp(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Parch(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Ticket(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Cabin(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "":
            dst[i] = -1
        else:
            dst[i] = val
    return dst


def prepro_Embarked(data):
    dst = np.zeros(data.shapes)
    for i, val in enumerate(data):
        if val == "Q":
            dst[i] = 0
        if val == "C":
            dst[i] = 0.5
        if val == "S":
            dst[i] = 1
    return dst
