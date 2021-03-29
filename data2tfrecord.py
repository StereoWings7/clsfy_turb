#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def feature_float_list(l):
    return tf.train.Feature(float_list=tf.train.FloatList(value=l))

def record2example(r):
    return tf.train.Example(features=tf.train.Features(feature={
        "x": feature_float_list(r[0:-1]),
        "y": feature_float_list([r[-1]])
    }))

filename_train_LR = "trainLR.tfrecords"
filename_train_HR = "trainHR.tfrecords"
filename_test_LR  = "testLR.tfrecords"
filename_test_HR  = "testHR.tfrecords"

ph_npz = ph_npz = np.load('diff-2force1_psi10_HRLR.npz')
x_train = ph_npz['ph_LR'][1:-50]
x_test  = ph_npz['ph_LR'][-50:]
y_train = ph_npz['ph_HR'][1:-50]
y_test  = ph_npz['ph_HR'][-50:]

# 前処理をする
x_train = x_train[...,np.newaxis].astype('float32')
x_test  = x_test[...,np.newaxis].astype('float32')
#x_test  = tf.Variable(x_test[...,tf.newaxis],dtype='float32')
# ラベルもfloat32型にする
y_train = y_train[...,np.newaxis].astype('float32')
y_test  = y_test[...,np.newaxis].astype('float32')
#y_test  = y_test.reshape((-1, 1)).astype("float32")
# TFRecord化するために、特徴量とラベルを結合する
#data_train = np.c_[x_train, y_train]

# 実際には、学習したいデータを同じ形式に変換して作る。
# 全データがメモリに乗り切らない場合は、以下の書き込みフェーズで
# 少しずつ作って書き込むことを繰り返せばよい。

# 学習データをTFRecordに書き込む
with tf.io.TFRecordWriter(filename_train_LR) as writer:
    for r in x_train:
        example = tf.train.Example(
            features = tf.train.Features(feature={
            "data": tf.train.Feature(float_list=tf.train.FloatList(value=r.flatten())),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=r.shape))
             }))
        writer.write(example.SerializeToString())

with tf.io.TFRecordWriter(filename_train_HR) as writer:
    for r in y_train:
        example = tf.train.Example(
            features = tf.train.Features(feature={
            "data": tf.train.Feature(float_list=tf.train.FloatList(value=r.flatten())),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=r.shape))
             }))
        writer.write(example.SerializeToString())


# 評価データをTFRecordに書き込む
with tf.io.TFRecordWriter(filename_test_LR) as writer:
    for r in x_test:
        example = tf.train.Example(
            features = tf.train.Features(feature={
            "data": tf.train.Feature(float_list=tf.train.FloatList(value=r.flatten())),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=r.shape))
             }))
        writer.write(example.SerializeToString())

with tf.io.TFRecordWriter(filename_test_HR) as writer:
    for r in y_test:
        example = tf.train.Example(
            features = tf.train.Features(feature={
            "data": tf.train.Feature(float_list=tf.train.FloatList(value=r.flatten())),
            "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=r.shape))
             }))
        writer.write(example.SerializeToString())

