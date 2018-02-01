import tensorflow as tf
import keras.backend as K
import numpy as np


def rmse(y_true, y_pred):
    rmse = K.sqrt(K.mean(K.square((y_true - y_pred))))
    return rmse


def of_loss(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) +0.000001)


def vo_loss(y_true, y_pred):
    mean = K.square(y_pred - y_true)
    mean_trasl = mean[:, 3:6]
    mean_rot = mean[:, 0:3] * 50
    mean = K.concatenate([mean_rot, mean_trasl])
    sqrt = K.sqrt(K.mean(mean, axis=-1))
    return sqrt


def rmse_loss(y_true, y_pred):
    mean = K.square(y_pred - y_true)
    mean_trasl = mean[:, 3:6]
    mean_rot = mean[:, 0:3] * 50
    mean = K.concatenate([mean_rot, mean_trasl])
    sqrt = K.sqrt(K.mean(mean, axis=-1))
    return sqrt

def rmse_error(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.square((y_true - y_pred))))
    return rmse

