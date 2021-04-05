import numpy as np
import tensorflow as tf
import ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Input

# カスタムの損失関数のPython関数を実装


def evaluate_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def get_mse_loss(input_hr, input_gen):
    mse_hr_mean2 = tf.reduce_mean(tf.reduce_mean(
        tf.square(input_hr), axis=1, keepdims=True), axis=2, keepdims=True)
    mse_loss = tf.reduce_mean(tf.square(input_gen-input_hr)/mse_hr_mean2)
    return mse_loss


def get_tke_loss(input_hr, input_gen, dx, dy):
    tke_gen = ops.get_TKE(input_gen, dx, dy)
    tke_hr = ops.get_TKE(input_hr, dx, dy)
    tke_hr_mean2 = tf.reduce_mean(tf.reduce_mean(
        tf.square(tke_hr), axis=0, keepdims=True), axis=1, keepdims=True)
    tke_loss = tf.reduce_mean(tf.square(tke_gen-tke_hr)/tke_hr_mean2)
    return tke_loss


def get_ens_loss(input_hr, input_gen, dx, dy):
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

    def call(self, psi_hr, psi_gen):
        vel_gen = ops.get_velocity(psi_gen, self.dx, self.dy)
        vel_hr = ops.get_velocity(psi_hr, self.dx, self.dy)
        vel_grad_gen = ops.get_velocity_grad(vel_gen, self.dx, self.dy)

        mse_loss = get_mse_loss(psi_hr, psi_gen)
        ens_loss = get_ens_loss(psi_hr, psi_gen, self.dx, self.dy)
        continuity_loss = ops.get_continuity_residual(vel_grad_gen)

        content_loss = (1-self.lambda_ens) * mse_loss + \
            self.lambda_ens * ens_loss
        gen_loss = (1-self.lambda_phys) * content_loss + \
            self.lambda_phys * continuity_loss
        return gen_loss

# Super-resolution Image Generator


class GeneratorModel(Model):
    def __init__(self, input_shape, dx, dy, lambda_ens, lambda_phys):
        super().__init__()
        # how to instantiate this class:
        # lr_shape = lr_imgs[0].shape // lr_imgs[N][:,:] are batched images.
        # self.generator = Generator(lr_shape)
        input_shape = (input_shape[0], input_shape[1], 1)
        self.input_layer = Input(input_shape)
        self.loss = ResNetLoss(dx, dy, lambda_ens, lambda_phys)
        #self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        #self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # Pre stage(Down Sampling)
        self.pre = [
            ops.PeriodicPadding2D(4),
            Conv2D(64, kernel_size=9, strides=1, padding='valid'),
            # padding="valid", input_shape=input_shape),
            # 2021-3-23(Tue) 元論文(Ledig2017)だとReLuじゃなくてPReLuだが
            Activation(tf.nn.relu)
        ]
        # Residual Block
        self.res = [
            [ops.Res_block(64) for _ in range(7)]  # 7だとメモリ不足?
        ]
        # Middle stage
        self.middle = [
            ops.PeriodicPadding2D(1),
            Conv2D(64, kernel_size=3, strides=1, padding='valid'),
            BatchNormalization()
        ]
        # Pixel Shuffle(Up Sampling)
        self.ps = [
            [ops.Pixel_shuffler(upsample=2) for _ in range(2)],
            ops.PeriodicPadding2D(4),
            # Conv2D(3, kernel_size=9, strides=4,
            # 出力フィルター(upsamplingでgenerateしたpsiの値に)は1つ
            Conv2D(1, kernel_size=9, padding="valid", activation="tanh")
        ]

        self.out = self.call(self.input_layer)

    # def build(self, batch_input_shape):
    #    self.model = self.create_model(batch_input_shape.as_list())
    #    super().build(batch_input_shape)

    # def create_model(self, input_shape):
    def call(self, inputs):
        # Pre stage
        #inputs = tf.keras.layers.Input(input_shape)
        pre = inputs
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
        #middle = Add()([middle, pre])

        # Pixel Shuffle
        out = middle
        for layer in self.ps:
            if isinstance(layer, list):
                for l in layer:
                    out = l(out)
            else:
                out = layer(out)
        return out

    # def call(self, inputs):
    #    return self.model(inputs)

    # なくてもエラーは出ないが、訓練・テスト間、エポックの切り替わりで
    # トラッカーがリセットされないため、必ずmetricsのプロパティをオーバーライドすること
    # self.reset_metrics()はこのプロパティを参照している
#    @property
#    def metrics(self):
#        return self.loss_tracker

    def train_step(self, dataset):
        image_LR, image_HR = dataset

        with tf.GradientTape() as tape:
            image_SR = self(image_LR, training=True)
            loss_val = tf.reduce_mean(self.loss(image_HR, image_SR))
        grads = tape.gradient(loss_val, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # self.loss_tracker.update_state(loss_val)
        #mae_metric.update_state(image_HR, image_SR)
        self.compiled_metrics.update_state(image_HR, image_SR)
        return{m.name: m.result() for m in self.metrics}

    def test_step(self, dataset):
        image_LR, image_HR = dataset

        image_SR = self.model(image_LR)
        loss_val = tf.reduce_mean(self.loss(image_HR, image_SR))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
