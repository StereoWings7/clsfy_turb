import numpy as np
import tensorflow as tf
import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization

# カスタムの損失関数のPython関数を実装


def evaluate_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# # TensorFlow低水準APIではなく、Keras backend APIを使ってもOK
# import tensorflow.keras.backend as K
# # 上記のインポートを行ったうえで、以下の内容で上記のコードを書き換えるだけ
# # 「tf.reduce_mean」 → 「K.mean」
# # 「tf.square」 → 「K.square」

# カスタムの損失関数クラスを実装（レイヤーのサブクラス化）
# （tf.keras.losses.MeanSquaredError()の代わり）


def get_mse_loss(input_hr, input_gen):
    mse_hr_mean2 = tf.reduce_mean(tf.reduce_mean(
        tf.square(input_hr), axis=1, keepdims=True), axis=2, keepdims=True)
    mse_loss = tf.reduce_mean(tf.square(input_gen-input_hr)/mse_hr_mean2)
    return mse_loss


def get_tke_loss(input_hr, input_gen):
    tke_gen = ops.get_TKE(input_gen, dx, dy)
    tke_hr = ops.get_TKE(input_hr, dx, dy)
    tke_hr_mean2 = tf.reduce_mean(tf.reduce_mean(
        tf.square(tke_hr), axis=0, keepdims=True), axis=1, keepdims=True)
    tke_loss = tf.reduce_mean(tf.square(tke_gen-tke_hr)/tke_hr_mean2)
    return tke_loss


def get_ens_loss(input_hr, input_gen):
    vel_grad_gen = ops.get_velocity_grad(
        ops.get_velocity(input_gen, dx, dy), dx, dy)
    vel_grad_hr = ops.get_velocity_grad(
        ops.get_velocity(input_hr, dx, dy), dx, dy)
    vorticity_gen = ops.get_vorticity(vel_grad_gen)
    vorticity_hr = ops.get_vorticity(vel_grad_hr)
    ens_gen = ops.get_enstrophy(vorticity_gen)
    ens_hr = ops.get_enstrophy(vorticity_hr)
    ens_hr_mean2 = tf.reduce_mean(tf.reduce_mean(
        tf.square(ens_hr), axis=1, keepdims=True), axis=2, keepdims=True)
    ens_loss = tf.reduce_mean(tf.square(ens_gen-ens_hr)/ens_hr_mean2)
    return ens_loss


class ResNetLoss(tf.keras.losses.Loss):
    def __init__(self, dx, dy, lambda_ens, lambda_phys, name="custom_loss", **kwargs):
        super(ResNetLoss, self).__init__(name=name, **kwargs)
        self.dx = dx
        self.dy = dy
        self.lambda_ens = lambda_ens
        self.lambda_phys = lambda_phys

    def call(self, psi_gen, psi_hr):
        vel_gen = ops.get_velocity(psi_gen, dx, dy)
        vel_hr = ops.get_velocity(psi_hr, dx, dy)
        vel_grad_gen = ops.get_velocity_grad(vel_gen, dx, dy)

        mse_loss = get_mse_loss(psi_hr, psi_gen)
        ens_loss = get_ens_loss(psi_hr, psi_gen)
        continuity_loss = ops.get_continuity_residual(vel_grad_gen)

        content_loss = (1-self.lambda_ens) * mse_loss + \
            self.lambda_ens * ens_loss
        gen_loss = (1-self.lambda_phys) * content_loss + \
            self.lambda_phys * continuity_loss
        return gen_loss

# Super-resolution Image Generator


class Generator(Model):
    def __init__(self, input_shape):
        super().__init__()
        # how to instantiate this class:
        # lr_shape = lr_imgs[0].shape // lr_imgs[N][:,:] are batched images.
        # self.generator = Generator(lr_shape)
        #input_shape_test = (input_shape[0], input_shape[1], 4)
        input_shape = (input_shape[0], input_shape[1], 64)

        # Pre stage(Down Sampling)
        self.pre = [
            ops.PeriodicPadding2D(4),
            Conv2D(64, kernel_size=9, strides=1,
                   padding='valid', input_shape=input_shape),
            # padding="valid", input_shape=input_shape),
            # 2021-3-23(Tue) 元論文(Ledig2017)だとReLuじゃなくてPReLuだが
            Activation(tf.nn.relu)
        ]
        # Residual Block
        self.res = [
            [ops.Res_block(64, input_shape) for _ in range(7)]  # 7だとメモリ不足?
        ]
        # Middle stage
        self.middle = [
            ops.PeriodicPadding2D(1),
            Conv2D(64, kernel_size=3, strides=1,
                   padding='valid', input_shape=input_shape),
            BatchNormalization()
        ]
        # Pixel Shuffle(Up Sampling)
        self.ps = [
            [ops.Pixel_shuffler(128, input_shape) for _ in range(2)],
            ops.PeriodicPadding2D(4),
            # Conv2D(3, kernel_size=9, strides=4,
            # 出力フィルター(upsamplingでgenerateしたpsiの値に)は1つ
            Conv2D(1, kernel_size=9,
                   padding="valid", activation="tanh")
        ]

    def call(self, x):
        # Pre stage
        pre = x
        for layer in self.pre:
            pre = layer(pre)

        # Residual Block
        res = pre
        for layer in self.res:
            for l in layer:
                res = l(res)

        # Middle stage
        middle = res
        for layer in self.middle:
            middle = layer(middle)
        # Skip connection
        middle += pre

        # Pixel Shuffle
        out = middle
        for layer in self.ps:
            if isinstance(layer, list):
                for l in layer:
                    out = l(out)
            else:
                out = layer(out)

        return out
