import tensorflow as tf
import numpy as np
import ResNet

N_EPOCHS = 5
BATCH_SIZE = 16
TRAIN_HR_FILE = 'diff-2force1_psi10_HRLR.npz'


def main():
    ph_npz = np.load(TRAIN_HR_FILE)
    # initial data from HMEq_FFTW is filled w/ zeros (reason unknown)
    # to avoid picking up this mode, 0th arg starts from 1.(2021-3-26(Fri))
    # ↑this is just a dirty makeshift. data refinement should come first.
    ph_data_LR = tf.Variable(ph_npz["ph_LR"][1:, :, :, None], dtype=tf.float32)
    ph_data_HR = tf.Variable(ph_npz["ph_HR"][1:, :, :, None], dtype=tf.float32)
    dataset_LR = tf.data.Dataset.from_tensor_slices(
        ph_data_LR).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
    dataset_HR = tf.data.Dataset.from_tensor_slices(
        ph_data_HR).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
    dataset = tf.data.Dataset.zip((dataset_LR, dataset_HR))

#   physical parameters of low resolution images
    nx, ny, lx, ly = 128, 128, 300, 300
    dx, dy = lx/nx, ly/ny
    lambda_ens, lambda_phys = 0.125, 0.125
    ph_shape = (ph_data_LR[0].shape[0], ph_data_LR[0].shape[1])
    generator = ResNet.GeneratorModel(
        ph_shape, dx, dy, lambda_ens, lambda_phys)
    loss = ResNet.ResNetLoss(dx, dy, lambda_ens, lambda_phys)
    optim = tf.keras.optimizers.Adam()
    optim.lr = 0.01
    #acc = tf.keras.metrics.MSE()

    def train_on_batch(image_LR, image_HR):
        # forward path
        with tf.GradientTape() as tape:
            image_SR = generator(image_LR)
            loss_val = loss(image_HR, image_SR)
            tf.print(loss_val)
        # backward propagation
        gradients = tape.gradient(loss_val, generator.trainable_weights)
        # step optimizer
        optim.apply_gradients(zip(gradients, generator.trainable_weights))
        # acc.update_state(image_SR,image_HR)
        return loss_val

    for i in range(BATCH_SIZE):
        # acc.reset_states()
        print("Epoch=", i)
        for image_LR, image_HR in dataset:
            loss_val = train_on_batch(image_LR, image_HR)
        #train_acc =  acc.result().numpy
        if tf.math.is_nan(loss_val):
            print("NaN detected!")
            break

        print("Loss={}".format(loss_val))


if __name__ == "__main__":
    main()
