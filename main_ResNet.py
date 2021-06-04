import tensorflow as tf
import numpy as np
import ResNet

N_EPOCHS = 20
BATCH_SIZE = 4


def parse_example(example, length):
    features = tf.io.parse_single_example(
        example,
        features={
            # リストを読み込む場合は次元数を指定する
            "data":  tf.io.FixedLenFeature([length], dtype=tf.float32),
            #次元;(バッチ、nx,ny)の3次元 read.pyとdata2tfrecord.pyを参照
            "shape": tf.io.FixedLenFeature([3], dtype=tf.int64)
        })
    shape = features["shape"]
    x = tf.reshape(features["data"], shape)
    return x


def main():
    #ph_npz = np.load(TRAIN_HR_FILE)
    # initial data from HMEq_FFTW is filled w/ zeros (reason unknown)
    # to avoid picking up this mode, 0th arg starts from 1.(2021-3-26(Fri))
    # ↑this is just a dirty makeshift. data refinement should come first.
    #ph_data_LR = tf.Variable(ph_npz["ph_LR"][1:, :, :, None], dtype=tf.float32)
    #ph_data_HR = tf.Variable(ph_npz["ph_HR"][1:, :, :, None], dtype=tf.float32)
    # dataset_LR = tf.data.Dataset.from_tensor_slices(
    #    ph_data_LR).shuffle((BATCH_SIZE*N_EPOCHS-1)//2+1).batch(BATCH_SIZE)
    # dataset_HR = tf.data.Dataset.from_tensor_slices(
    #    ph_data_HR).shuffle((BATCH_SIZE*N_EPOCHS-1)//2+1).batch(BATCH_SIZE)
    #dataset = tf.data.Dataset.zip((dataset_LR, dataset_HR))
    dataset_LR = tf.data.TFRecordDataset(["trainLR.tfrecords"]).map(
        lambda x: parse_example(x, 128*128))
    dataset_HR = tf.data.TFRecordDataset(["trainHR.tfrecords"]).map(
        lambda x: parse_example(x, 512*512))
    dataset = tf.data.Dataset.zip((dataset_LR, dataset_HR)).batch(
        BATCH_SIZE, drop_remainder=True).shuffle(N_EPOCHS*BATCH_SIZE).prefetch(1)

#   physical parameters of low resolution images
    nx, ny, lx, ly = 128, 128, 300, 300
    dx, dy = lx/nx, ly/ny
    lambda_ens, lambda_phys = 0.125, 0.125
    ph_shape = (nx, ny)
    generator = ResNet.GeneratorModel(
        ph_shape, dx, dy, lambda_ens, lambda_phys)
    loss = ResNet.ResNetLoss(dx, dy, lambda_ens, lambda_phys)
    optim = tf.keras.optimizers.Adam(lr=1e-3)
    loss_tracker = tf.keras.metrics.Mean(name='loss')
    #acc = tf.keras.metrics.MSE()

    @tf.function
    def train_on_batch(image_LR, image_HR):
        # forward path
        with tf.GradientTape() as tape:
            image_SR = generator(image_LR)
            loss_val = loss(image_HR, image_SR)
            # tf.print(loss_val)
        # backward propagation
        gradients = tape.gradient(loss_val, generator.trainable_weights)
        # step optimizer
        optim.apply_gradients(zip(gradients, generator.trainable_weights))
        loss_tracker(loss_val)
        # acc.update_state(image_SR,image_HR)
        return loss_val

    for i in range(N_EPOCHS):
        # acc.reset_states()
        print("Epoch=", i)
        for image_LR, image_HR in dataset:
            loss_val = train_on_batch(image_LR, image_HR)
        #train_acc =  acc.result().numpy
        if tf.math.is_nan(loss_val):
            print("NaN detected!")
            break

        print("Loss={}".format(loss_val))

    generator.save("my_custom_model.ckpt")
    iter_val = iter(tf.data.TFRecordDataset(["trainLR.tfrecords"]).map(
        lambda x: parse_example(x, 128*128)).batch(1))
    val_LR = iter_val.get_next()
    val_SR = generator(val_LR).numpy()
    np.savez('image_SR', image_SR=val_SR)


if __name__ == "__main__":
    main()
