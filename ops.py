import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.activations import get
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Add, Activation, Dense, Flatten, BatchNormalization, Conv2D, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import backend as K


class PeriodicPadding2D(Layer):

    def __init__(self, padding=1, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.padding = conv_utils.normalize_tuple(padding, 1, 'padding')
        self.input_spec = InputSpec(ndim=4)

    def wrap_pad(self, input, size):
        M1 = tf.concat([input[:, :, -size:, :], input,
                        input[:, :, 0:size, :]], 2)
        M1 = tf.concat([M1[:, -size:, :, :], M1, M1[:, 0:size, :, :]], 1)
        return M1

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 4
        if shape[1] is not None:
            length = shape[1] + 2*self.padding[0]
        else:
            length = None
        return tuple([shape[0], length, length, shape[3]])

    def call(self, inputs):
        return self.wrap_pad(inputs, self.padding[0])

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(PeriodicPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Pixel Shuffle Block


def pixel_shuffler(inputs, shuffle_stride=2):
    batch_size = tf.shape(inputs)[0]
    _, H, W, C = inputs.get_shape()
    r1 = shuffle_stride
    r2 = shuffle_stride
    out_c = C//(r1*r2)
    out_h = H * r1
    out_w = W * r2

    assert C == r1 * r2 * out_c

    x = tf.reshape(inputs, (batch_size, H, W, r1, r2, out_c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (batch_size, out_h, out_w, out_c))

    return x


class Pixel_shuffler(Layer):
    def __init__(self, upsample=2, **kwargs):
        self.upsample = upsample
        super(Pixel_shuffler, self).__init__(**kwargs)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def compute_output_shape(self, batch_input_shape):
        shape = list(batch_input_shape)
        assert len(shape) == 4
        if shape[1] is not None:
            length = shape[1] * self.upsample
        else:
            length = None
        return tuple([shape[0], length, length, shape[3]/(self.upsample**2)])

    def call(self, inputs):
        return pixel_shuffler(inputs, shuffle_stride=self.upsample)

    def get_config(self):
        config = {'upsample': self.upsample}
        base_config = super(Pixel_shuffler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Residual Block


class Res_block(Layer):
    def __init__(self, ch):
        super().__init__()
        # ToDo 2021/03/23: input periodic_padding layer before conv2d
        self.perd_pad1 = PeriodicPadding2D(1)
        self.conv1 = Conv2D(ch, kernel_size=3, strides=1,
                            padding="valid", kernel_initializer="he_normal")
        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.perd_pad2 = PeriodicPadding2D(1)
        self.conv2 = Conv2D(ch, kernel_size=3, strides=1,
                            padding="valid", kernel_initializer="he_normal")
        self.bn2 = BatchNormalization()
        self.add = Add()

    def call(self, x):
        d1 = self.av1(self.bn1(self.conv1(self.perd_pad1(x))))
        d2 = self.bn2(self.conv2(self.perd_pad2(d1)))
        return self.add([x, d2])


class Conv_Discriminator(Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = get(activation)
        self.main_layers = [
            PeriodicPadding2D(1),
            Conv2D(filters, kernel_size=3, strides=strides),
            BatchNormalization(),
            self.activation
        ]

    def call(self, x):
        for layer in self.main_layers:
            x = layer(x)
        return x

# derivative of velocity
# dim channel: distinguish u,v, and w (not psi itself!)


def ddx(input, dx, name=None):
    input_shape = input.get_shape().as_list()

    # derivative implemented by conv layer
    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
                         4., -3./20., 1./60.], dtype=tf.float32)
    # conv filter shape: last 2 dim: # of input/output filter
    # for compt. deriv., # of both in/out is 1.
    ddx2D = tf.reshape(ddx1D, shape=(-1, 1, 1, 1))

    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
    strides = [1, 1, 1, 1]
    # 2021-3-25(Thu) caution!! func periodic_padding w/ current implmt. cannot be executed in graph mode.
    # as a makeshift, simple zero padding is applied here
    # this change shouldn't affect the result well cuz its only used to give loss func
    #var_pad_pre = periodic_padding(input, input_shape, 3)
    # expand axis to conform w/ conv layer
    #var_pad = var_pad_pre[:, :, 1:input_shape[2]+1, :]

    # output = tf.nn.conv2d(var_pad, ddx2D, strides,
    input_perd = PeriodicPadding2D(padding=3)(input)[:, :, 3:-3, :]
    output = tf.nn.conv2d(input_perd, ddx2D, strides,
                          padding='SAME', name=name)
    output = tf.scalar_mul(1./dx, output)
    return output[:, 3:-3, :, :]


def ddy(input, dy, name=None):
    input_shape = input.get_shape().as_list()

    # derivative implemented by conv layer
    ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
                         4., -3./20., 1./60.], dtype=tf.float32)
    # conv filter shape: last 2 dim: # of input/output filter
    # for compt. deriv., # of both in/out is 1.
    ddy2D = tf.reshape(ddy1D, shape=(1, -1, 1, 1))

    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
    strides = [1, 1, 1, 1]
    # 2021-3-25(Thu) caution!! func periodic_padding w/ current implmt. cannot be executed in graph mode.
    # as a makeshift, simple zero padding is applied here
    # this change shouldn't affect the result well cuz its only used to give loss func
    #var_pad_pre = periodic_padding(input, input_shape, 3)
    # expand axis to conform w/ conv layer
    #var_pad = var_pad_pre[:, 1:input_shape[1]+1, :, :]

    #output = tf.nn.conv2d(var_pad, ddy2D, strides, padding='VALID', name=name)
    input_perd = PeriodicPadding2D(padding=3)(input)[:, 3:-3, :, :]
    output = tf.nn.conv2d(input_perd, ddy2D, strides,
                          padding='SAME', name=name)
    output = tf.scalar_mul(1./dy, output)
    return output[:, :, 3:-3, :]

# seconde derivative of velocity
# dim channel: distinguish u,v, and w (not psi itself!)

# use w/ caution! periodic_padding may cause error.
# def d2dx2(input, channel, dx, name=None):
#    input_shape = input.get_shape().as_list()
#    # expand axis to conform w/ conv layer
#    var = tf.expand_dims(input[:, :, :, channel], axis=0)
#
#    # derivative implemented by conv layer
#    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
#                         4., -3./20., 1./60.], dtype=tf.float32)
#    # conv filter shape: last 2 dim: # of input/output filter
#    # for compt. deriv., # of both in/out is 1.
#    ddx2D = tf.reshape(ddx1D, shape=(-1, 1, 1, 1))
#
#    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
#    strides = [1, 1, 1, 1]
#    var_pad_pre = periodic_padding(input, input_shape, 3)
#    # expand axis to conform w/ conv layer
#    var_pad = var_pad_pre[:, :, 1:input_shape[2]+1, :]
#
#    output = tf.nn.conv2d(var_pad, ddx2D, strides, padding='VALID', name=name)
#    output = tf.scalar_mul(1./dx**2, output)
#    return output
#
#
# def d2dy2(input, channel, dy, name=None):
#    input_shape = input.get_shape().as_list()
#    # expand axis to conform w/ conv layer
#    var = tf.expand_dims(input[:, :, :, channel], axis=3)
#
#    # derivative implemented by conv layer
#    ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
#                         4., -3./20., 1./60.], dtype=tf.float32)
#    # conv filter shape: last 2 dim: # of input/output filter
#    # for compt. deriv., # of both in/out is 1.
#    ddy2D = tf.reshape(ddy1D, shape=(1, -1, 1, 1))
#
#    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
#    strides = [1, 1, 1, 1]
#    var_pad_pre = periodic_padding(input, input_shape, 3)
#    # expand axis to conform w/ conv layer
#    var_pad = var_pad_pre[:, 1:input_shape[1]+1, :, :]
#    output = tf.nn.conv2d(var_pad, ddy2D, strides, padding='VALID', name=name)
#    output = tf.scalar_mul(1./dy**2, output)
#    return output


def get_velocity(psi, dx, dy, name=None):
    dpdx = ddx(psi, dx)
    dpdy = ddy(psi, dy)
    return dpdy, -dpdx


def get_TKE(psi, dx, dy, name=None):
    u, v = get_velocity(psi, dx, dy)
    TKE = 0.5*(tf.square(u)+tf.square(v))
    return TKE


def get_velocity_grad(vel_list, dx, dy, name=None):
    u, v = vel_list
    dudx = ddx(u, dx)
    dudy = ddy(u, dy)
    dvdx = ddx(v, dx)
    dvdy = ddy(v, dy)
    return dudx, dudy, dvdx, dvdy


def get_strain_rate_mag2(vel_grad, name=None):
    dudx, dvdx, dudy, dvdy, = vel_grad
    strain_rate_mag2 = dudx**2 + dvdy**2 + 2*((0.5*(dudy + dvdx))**2)
    return strain_rate_mag2


def get_vorticity(vel_grad, name=None):
    _, dudy, dvdx, _ = vel_grad
    vort = dvdx - dudy
    return vort


def get_enstrophy(vorticity, name='enstrophy'):
    omega = vorticity
    ens = omega**2
    return ens


def get_continuity_residual(vel_grad, name='continiuity'):
    dudx, _, _, dvdy = vel_grad
    res = dudx + dvdy
    res = tf.reduce_mean(tf.reduce_mean(
        dudx, axis=1, keepdims=True), axis=2, keepdims=True)+tf.reduce_mean(tf.reduce_mean(
            dvdy, axis=1, keepdims=True), axis=2, keepdims=True)
    return res
