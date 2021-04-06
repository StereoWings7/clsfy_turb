import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import ops
import ResNet

nx, ny, lx, ly = 128, 128, 300, 300
dx, dy = lx/nx, ly/ny
dkx, dky = np.pi/lx, np.pi/ly
xx, yy = np.linspace(0, lx, nx), np.linspace(0, ly, ny)
XX, YY = np.meshgrid(xx, yy)
xx_HR, yy_HR = np.linspace(0, lx, nx*4), np.linspace(0, ly, ny*4)
XX_HR, YY_HR = np.meshgrid(xx_HR, yy_HR)
# 適当に試験関数をつくる
test_sin = np.sin(3*dkx*XX+3*dky*YY)
# 試験関数をtensorflow objectに変換
test_tf_sin = tf.convert_to_tensor(
    test_sin[np.newaxis, ..., np.newaxis], dtype=tf.float32)

# 試験関数じゃなくて直にデータからとってくる

BATCH_SIZE, N_EPOCHS = 16, 10
lambda_ens, lambda_phys = 0.25, 0.25
ph_shape = (nx, ny, 1)


def parse_example(example, length):
    features = tf.io.parse_single_example(
        example,
        features={
            # リストを読み込む場合は次元数を指定する
            "data":  tf.io.FixedLenFeature([length], dtype=tf.float32),
            "shape": tf.io.FixedLenFeature([3], dtype=tf.int64)
        })
    shape = features["shape"]
    x = tf.reshape(features["data"], shape)
    return x


dataset_LR = tf.data.TFRecordDataset(["trainLR.tfrecords"]).map(lambda x: parse_example(
    x, 128*128)).batch(BATCH_SIZE, drop_remainder=True).shuffle(N_EPOCHS*BATCH_SIZE).prefetch(1)
dataset_HR = tf.data.TFRecordDataset(["trainHR.tfrecords"]).map(lambda x: parse_example(
    x, 512*512)).batch(BATCH_SIZE, drop_remainder=True).shuffle(N_EPOCHS*BATCH_SIZE).prefetch(1)
dataset = tf.data.Dataset.zip((dataset_LR, dataset_HR))

# generatorモデルから作ってみる
ph_LR_singlebatch, ph_HR_singlebatch = next(iter(dataset))
ph_LR_single = tf.expand_dims(ph_LR_singlebatch[0, ...], 0)
ph_HR_single = tf.expand_dims(ph_HR_singlebatch[0, ...], 0)
generator = tf.keras.models.load_model('my_custom_model.ckpt')
ph_SR_single = generator(ph_LR_single)

# 試験関数をops.pyで作ったddx層に代入してみる
#test_tf_sin_dx = ops.ddx(test_tf_sin, dx)
ph_LR_dx = ops.ddx(ph_LR_single, dx)
ph_LR_dy = ops.ddy(ph_LR_single, dy)
# 可視化
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131)
#im1 = ax1.imshow(test_tf_sin[0, ..., 0], cmap='jet')
im1 = ax1.imshow(ph_LR_single[0, ..., 0], cmap='jet')
fig.colorbar(im1)
ax2 = fig.add_subplot(132)
im2 = ax2.imshow(ph_LR_dx[0, ..., 0], cmap='jet')
fig.colorbar(im2)
ax3 = fig.add_subplot(133)
im3 = ax3.imshow(ph_LR_dy[0, ..., 0], cmap='jet')
fig.colorbar(im3)
fig.savefig('ddx_TF_test.png')

# 今度はperiodic padding層を試す
#test_tf_sin_ps = ops.PeriodicPadding2D(padding=4)(test_tf_sin)
ph_LR_ps = ops.PeriodicPadding2D(padding=24)(ph_LR_single)
fig.clear()
# 可視化
fig = plt.figure()
ax1 = fig.add_subplot(121)
#im1 = ax1.imshow(test_tf_sin[0, ..., 0], cmap='jet')
im1 = ax1.imshow(ph_LR_single[0, ..., 0], cmap='jet')
fig.colorbar(im1)
ax2 = fig.add_subplot(122)
#im2 = ax2.imshow(test_tf_sin_ps[0, ..., 0], cmap='jet')
im2 = ax2.imshow(ph_LR_ps[0, ..., 0], cmap='jet')
fig.colorbar(im2)
fig.savefig('ps_TF_test.png')

# 速度場も出してみる
vx, vy = ops.get_velocity(ph_LR_single, dx, dy)
# 可視化
fig.clear()
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131)
#im1 = ax1.imshow(test_tf_sin[0, ..., 0], cmap='jet')
im1 = ax1.imshow(ph_LR_single[0, ..., 0], cmap='jet')
fig.colorbar(im1)
ax2 = fig.add_subplot(132)
#im2 = ax2.imshow(test_tf_sin_ps[0, ..., 0], cmap='jet')
im2 = ax2.imshow(vx[0, ..., 0], cmap='jet')
fig.colorbar(im2)
ax3 = fig.add_subplot(133)
im3 = ax3.imshow(vy[0, ..., 0], cmap='jet')
fig.colorbar(im3)
fig.savefig('vel_TF_test.png')


# 可視化
fig.clear()
fig = plt.figure()
ax1 = fig.add_subplot(121)
#im1 = ax1.imshow(test_tf_sin[0, ..., 0], cmap='jet')
im1 = ax1.imshow(ph_LR_single[0, ..., 0], cmap='jet')
fig.colorbar(im1)
ax2 = fig.add_subplot(122)
#im2 = ax2.imshow(test_tf_sin_ps[0, ..., 0], cmap='jet')
im2 = ax2.imshow(ph_SR_single[0, ..., 0], cmap='jet')
fig.colorbar(im2)
fig.savefig('SR_TF_test.png')
