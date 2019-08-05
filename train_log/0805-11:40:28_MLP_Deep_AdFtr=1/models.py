# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, MaxPool1D, Flatten, BatchNormalization
from keras import regularizers
from keras.utils import plot_model


def model_construct_mlp_shallow(train_data):
    inputs = Input(train_data.shape[1:])
    data = BatchNormalization()(inputs)
    data = Dense(30, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(50, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = BatchNormalization()(data)
    data = Dense(30, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(2, activation="softmax", kernel_initializer="random_uniform", bias_initializer="constant")(data)

    mlp = Model(inputs, data, name="MLP_shallow_1")
    mlp.summary()

    return mlp


def model_construct_mlp_deep(train_data):
    inputs = Input(train_data.shape[1:])
    data = BatchNormalization()(inputs)
    data = Dense(30, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(50, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = BatchNormalization()(data)
    data = Dense(100, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(100, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(50, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(30, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(2, activation="softmax", kernel_initializer="random_uniform", bias_initializer="constant")(data)

    mlp = Model(inputs, data, name="MLP_deep_1")
    mlp.summary()

    return mlp


def model_construct_mlp_deep_l2(train_data):
    inputs = Input(train_data.shape[1:])
    data = BatchNormalization()(inputs)
    data = Dense(30, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = Dense(50, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = BatchNormalization()(data)
    data = Dense(100, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = Dense(100, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = Dense(50, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = Dense(30, activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="constant",
                 kernel_regularizer=regularizers.l2(0.1))(data)
    data = Dense(2, activation="softmax", kernel_initializer="random_uniform", bias_initializer="constant")(data)

    mlp = Model(inputs, data, name="MLP_deep_2")
    mlp.summary()

    return mlp


def model_construct_cnn1d(train_data):
    inputs = Input(shape=train_data.shape[1:] + (1, ))
    data = Conv1D(312, 7, padding="same")(inputs)
    data = MaxPool1D()(data)
    data = Conv1D(128, 3)(data)
    data = Flatten()(data)
    data = Dense(20, activation="relu", kernel_initializer="random_uniform", bias_initializer="constant")(data)
    data = Dense(2, activation="softmax", kernel_initializer="random_uniform", bias_initializer="constant")(data)

    cnn1d = Model(inputs, data, name="CNN1D_1")
    cnn1d.summary()

    return cnn1d
