import tensorflow as tf
import TEGAN

N_EPOCHS = 20
BATCH_SIZE = 4


def parse_example(example, length):
    features = tf.io.parse_single_example(
        example,
        features={
            # リストを読み込む場合は次元数を指定する
            "data":  tf.io.FixedLenFeature([length], dtype=tf.float32),
            # 次元;(バッチ、nx,ny)の3次元 read.pyとdata2tfrecord.pyを参照
            "shape": tf.io.FixedLenFeature([3], dtype=tf.int64)
        })
    shape = features["shape"]
    x = tf.reshape(features["data"], shape)
    return x


def main():
    # serialize data
    dataset_LR = tf.data.TFRecordDataset(["trainLR.tfrecords"]).map(
        lambda x: parse_example(x, 128*128))
    dataset_HR = tf.data.TFRecordDataset(["trainHR.tfrecords"]).map(
        lambda x: parse_example(x, 512*512))
    dataset = tf.data.Dataset.zip((dataset_LR, dataset_HR)).batch(
        BATCH_SIZE, drop_remainder=True).shuffle(N_EPOCHS*BATCH_SIZE).prefetch(1)

#   input size = size of pictures of which TEGAN take as a input.
    nx, ny = 512, 512
    ph_shape = (nx, ny)

    # generatorは既にmain_ResNet.pyで学習済みのものをインポート,
    # discriminator はTEGAN.pyで作成したモデルを利用
    generator = tf.keras.models.load_model("my_custom_model.ckpt")
    discriminator = TEGAN.DiscriminatorModel(ph_shape)
    # 損失関数はGAN用の2値分類
    loss_TEGAN = tf.keras.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.Adam(lr=1e-3)
    loss_tracker = tf.keras.metrics.Mean(name='loss')
    #acc = tf.keras.metrics.MSE()

    @tf.function
    def train_discriminator(image_HR, y_true):
        with tf.GradientTape() as tape:
            y_pred = discriminator(image_HR)
            loss_val = loss_TEGAN(y_true, y_pred)
            # tf.print(loss_val)
        # backward propagation
        gradients = tape.gradient(loss_val, discriminator.trainable_weights)
        # step optimizer
        optim.apply_gradients(zip(gradients, discriminator.trainable_weights))
        loss_tracker(loss_val)
        # acc.update_state(image_SR,image_HR)
        return loss_val

    @tf.function
    def train_generator(image_LR, y_true):
        # forward path
        with tf.GradientTape() as tape:
            #loss_val = loss(image_HR, image_SR)
            y_pred = discriminator(generator(image_LR))
            loss_val = loss_TEGAN(y_true, y_pred)
            # tf.print(loss_val)
        # backward propagation
        gradients = tape.gradient(loss_val, generator.trainable_weights)
        # step optimizer
        optim.apply_gradients(zip(gradients, generator.trainable_weights))
        loss_tracker(loss_val)
        # acc.update_state(image_SR,image_HR)
        return loss_val

    # 先にmy_custom_model.ckptからResNetの重みを読み込んでおく。
    reconst_model = tf.keras.models.load_model("my_custom_model.ckpt")
    # あるいは、ここでまとめてResNetのほうの訓練もやっておく。

    for i in range(N_EPOCHS):
        # acc.reset_states()
        print("Epoch=", i)
        for image_LR, image_HR in dataset:
            # 1. train discriminator
            image_SR = reconst_model(image_LR)
            fake_and_real = tf.concat([image_SR, image_HR], axis=0)
            y1 = tf.constant([[0.]] * BATCH_SIZE + [[1.]] * BATCH_SIZE)
            loss_disc = train_discriminator(fake_and_real, y1)
            # 2. train generator
            y2 = tf.constant([[1.]]*BATCH_SIZE)
            loss_gen = train_generator(image_LR, y2)
        #train_acc =  acc.result().numpy
        if tf.math.is_nan(loss_disc) or tf.math.is_nan(loss_gen):
            print("NaN detected!")
            break
        print("Loss_disc={0}, Loss_gen={1}".format(loss_disc, loss_gen))

#    generator.save("my_custom_model.ckpt")
#    iter_val = iter(tf.data.TFRecordDataset(["trainLR.tfrecords"]).map(
#        lambda x: parse_example(x, 128*128)).batch(1))
#    val_LR = iter_val.get_next()
#    val_SR = generator(val_LR).numpy()
#    np.savez('image_SR', image_SR=val_SR)


if __name__ == "__main__":
    main()
