import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Add, Activation, Dense, Flatten, BatchNormalization, Conv2D
from tensorflow.python.keras import backend as K


# def periodic_padding(x, padding):
#    # dim 0 = batch size
#    d1 = x.get_shape().as_list()[1]
#    d2 = x.get_shape().as_list()[2]
#    # materializing temp. tensor 9x larger w/ orig. needed for periodic padding
#    dilated = tf.tile(x, [1, 3, 3, 1])
#    result = dilated[:, d1-padding:2*d1+padding, d2-padding:2*d2+padding, :]
#    return result
def periodic_padding(input, padding):
    d1 = input.get_shape().as_list()[1]
    d2 = input.get_shape().as_list()[2]
    d3 = input.get_shape().as_list()[3]
    x = []
    for input_each in input:
        padded = np.zeros((d1+2*padding, d2+2*padding, d3))
        # lower left corner
        padded[:padding, :padding, :] = input_each[d1-padding:, d2-padding:, :]
        # lower middle
        padded[padding:d1+padding, :padding, :] = input_each[:, d2-padding:, :]
        # lower right corner
        padded[d1+padding:, :padding, :] = input_each[:padding, d2-padding:, :]
        # left side
        padded[:padding, padding:d2+padding, :] = input_each[d1-padding:, :, :]
        # center
        padded[padding:d1+padding, padding:d2+padding, :] = input_each[:, :, :]
        # right side
        padded[d1+padding:, padding:d2+padding, :] = input_each[:padding, :, :]
        # top left corner
        padded[:padding, d2+padding:, :] = input_each[d1-padding:, :padding, :]
        # top middle
        padded[padding:d1+padding, d2+padding:, :] = input_each[:, :padding, :]
        # top right corner
        padded[d1+padding:, d2+padding:, :] = input_each[:padding, :padding, :]
        x.append(padded)
    return tf.Variable(x, dtype=tf.float32)


class PeriodicPadding2D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = padding
        super(PeriodicPadding2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return periodic_padding(x, self.padding)

# def periodic_padding(tensor, axis, pad_len=1):
#    shape = tensor.get_shape().as_list()
#    length = len(shape)
#    padding = np.zeros((0, 2))
#    for j in length:
#        paddings = np.vstack((paddings, np.array([[pad_len, pad_len]])))
#    paddings = tf.constant(paddings)
#    # result = tf.pad(tensor,paddings,"REFLECT")
#    return tf.pad(tensor, paddings, "REFLECT")


# def Conv2D_periodic_padding(input, filter, strides, name=None):
#    #input_shape = input.get_shape().as_list()
#    filter_shape = filter.get_shape().as_list()
#    pad = []
#
#    for i_dim in range(2):
#        # Computing pad width assuming output_size = input_size/stride and size of the #filter is odd.
#        padL = int(0.5*(filter_shape[i_dim]-1))
#        padR = padL
#        pad_idim = (pad_L, pad_R)
#        pad.append(pad_idim)
#
#    input_pad = periodic_padding(input, pad)
#    output = tf.nn.conv2d(input_pad, filter, strides,
#                          padding='valid', name=name)
#
#    return output

# Pixel Shuffle Block


class Pixel_shuffler(Layer):
    def __init__(self, out_ch, input_shape):
        super().__init__()
        self.conv = Conv2D(out_ch, kernel_size=3, strides=1,
                           padding="same", input_shape=input_shape)
        # 2021-3-23(Tue) ここ元論文(Ledig2017)だとReLuじゃなくてPReLuだが
        self.act = Activation(tf.nn.relu)

    # forward proc
    def call(self, x):
        d1 = self.conv(x)
        d2 = self.act(tf.nn.depth_to_space(d1, 2))
        return d2


# Residual Block


class Res_block(Layer):
    def __init__(self, ch, input_shape):
        super().__init__()
        # ToDo 2021/03/23: input periodic_padding layer before conv2d
        self.perd_pad1 = PeriodicPadding2D(1)
        self.conv1 = Conv2D(ch, kernel_size=3, strides=1,
                            padding="valid")
        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.perd_pad2 = PeriodicPadding2D(1)
        self.conv2 = Conv2D(ch, kernel_size=3, strides=1, padding="valid")
        self.bn2 = BatchNormalization()
        self.add = Add()

    def call(self, x):
        d1 = self.av1(self.bn1(self.conv1(self.perd_pad1(x))))
        d2 = self.bn2(self.conv2(self.perd_pad2(d1)))
        return self.add([x, d2])
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
    var_pad_pre = periodic_padding(input, 3)
    # expand axis to conform w/ conv layer
    var_pad = var_pad_pre[:, :, 1:input_shape[2]+1, :]

    output = tf.nn.conv2d(var_pad, ddx2D, strides,
                          padding='VALID', name=name)
    output = tf.scalar_mul(1./dx, output)
    return output


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
    var_pad_pre = periodic_padding(input, 3)
    # expand axis to conform w/ conv layer
    var_pad = var_pad_pre[:, 1:input_shape[1]+1, :, :]

    output = tf.nn.conv2d(var_pad, ddy2D, strides, padding='VALID', name=name)
    output = tf.scalar_mul(1./dy, output)
    return output

# seconde derivative of velocity
# dim channel: distinguish u,v, and w (not psi itself!)


def d2dx2(input, channel, dx, name=None):
    input_shape = input.get_shape().as_list()
    # expand axis to conform w/ conv layer
    var = tf.expand_dims(input[:, :, :, channel], axis=0)

    # derivative implemented by conv layer
    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
                         4., -3./20., 1./60.], dtype=tf.float32)
    # conv filter shape: last 2 dim: # of input/output filter
    # for compt. deriv., # of both in/out is 1.
    ddx2D = tf.reshape(ddx1D, shape=(-1, 1, 1, 1))

    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
    strides = [1, 1, 1, 1]
    var_pad_pre = periodic_padding(input, 3)
    # expand axis to conform w/ conv layer
    var_pad = var_pad_pre[:, :, 1:input_shape[2]+1, :]

    output = tf.nn.conv2d(var_pad, ddx2D, strides, padding='VALID', name=name)
    output = tf.scalar_mul(1./dx**2, output)
    return output


def d2dy2(input, channel, dy, name=None):
    input_shape = input.get_shape().as_list()
    # expand axis to conform w/ conv layer
    var = tf.expand_dims(input[:, :, :, channel], axis=3)

    # derivative implemented by conv layer
    ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3. /
                         4., -3./20., 1./60.], dtype=tf.float32)
    # conv filter shape: last 2 dim: # of input/output filter
    # for compt. deriv., # of both in/out is 1.
    ddy2D = tf.reshape(ddy1D, shape=(1, -1, 1, 1))

    # strides shape diff. w/ tf.nn.conv2d and tf.keras.layers.conv2d
    strides = [1, 1, 1, 1]
    var_pad_pre = periodic_padding(input, 3)
    # expand axis to conform w/ conv layer
    var_pad = var_pad_pre[:, 1:input_shape[1]+1, :, :]
    output = tf.nn.conv2d(var_pad, ddy2D, strides, padding='VALID', name=name)
    output = tf.scalar_mul(1./dy**2, output)
    return output


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
    return res
