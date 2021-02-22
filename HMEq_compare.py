import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy.core.numeric import full
from scipy.fft import fftshift, ifft2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


class HMEq_data:
    head = ("head", "<i")
    tail = ("tail", "<i")

    def __init__(self, nx=512, ny=512, lx=300, ly=300) -> None:
        pass
        self.nx, self.ny = nx, ny
        self.nkx, self.nky = (nx-2)//3, (ny-2)//3
        self.lx, self.ly = lx, ly
        self.dx, self.dy = lx/nx, ly/ny
        self.xx = np.linspace(0, lx, nx)
        self.yy = np.linspace(0, ly, ny)
        self.XX, self.YY = np.meshgrid(self.xx, self.yy)
        self.dt_pk = np.dtype(
            [self.head, ("pk", "{0}<c16".format((2*self.nkx+1)*(self.nky+1))), self.tail])
        self.dt_time = np.dtype([self.head, ("time", "<d"), self.tail])
        self.pk_size = self.dt_pk.itemsize
        self.time_size = self.dt_time.itemsize

    def __del__(self):
        pass

    def datafile_phi(self, ifrun, t_spc=100):
        ph_data = np.empty((0, (self.nx)*(self.ny)), dtype=float)
        iter_global = 0
        for ifile in np.arange(1, ifrun):
            fd_pk = open("data/pk_{0:03}.bin".format(ifile), "rb")
            fd_time = open("data/time_{0:03}.bin".format(ifile), "rb")
            chunk_time = np.fromfile(fd_time, dtype=self.dt_time, count=10000)
            for iter in range(chunk_time.size):
                iter_global += 1
                if(iter_global % t_spc == 0):
                    fd_pk.seek(iter*self.pk_size)
                    chunk_pk = np.fromfile(fd_pk, dtype=self.dt_pk, count=1)
                    pk_tmp = chunk_pk[0]["pk"].reshape(
                        (2*self.nkx+1, self.nky+1), order="F")
                    ph_tmp = self._fft_backward(pk_tmp).flatten()
                    ph_data = np.append(ph_data, np.array([ph_tmp]), axis=0)
        return ph_data

    def _fft_backward(self, pk):
        pk_lx = 2*self.nkx+1
        pk_ly = self.nky+1

        pk_full = np.zeros((pk_lx, pk_ly+self.nky), dtype='c16')
        ph = np.zeros((self.nx, self.ny), dtype='c16')

        # in HMEq_FFTW, normalization coeff is applied at forward transformation
        # (caution!! ↑that treatment is different from ordinal FFT procedure!!# )
        # realistic condition enforcing pattern
        coeff_norm = self.nx*self.ny

        for i in range(pk_lx):
            for j in range(pk_ly):
                pk_full[i, j+self.nky] = coeff_norm*pk[i, j]
        for i in range(pk_lx):
            for j in range(1, self.nky+1):
                ikx = i-self.nkx
                iky = j
                pk_full[i, j] = coeff_norm * \
                    pk[self.nkx-ikx, self.nky-iky].conjugate()

        # copy from pk_full to ph
        for i in range(pk_lx):
            for j in range(pk_ly+self.nky):
                ist = self.nx//2-self.nkx
                jst = self.ny//2-self.nky
                ph[i+ist, j+jst] = pk_full[i, j]
        phr = fftshift(ph)
        return np.real(ifft2(phr))


def gen_rand(batch, shape, seed=0):
    np.random.seed(seed)
    phi_rand = np.random.rand(batch, shape)
    return phi_rand, np.zeros(batch)


def sampling_down(ph_batch, ndx=128, ndy=128):
    nx = ph_batch.shape[1]
    ny = ph_batch.shape[2]
    xstep = nx//ndx
    ystep = ny//ndy
    ph_down = ph_batch[:, ::xstep, ::ystep]
    return ph_down


def full_connect(x):
    input_layer = Input(shape=(x.flatten().size,))
    layer2 = Dense(512, activation='relu')(input_layer)
    layer2 = Dropout(0.2)(layer2)
    layer3 = Dense(512, activation='relu')(layer2)
    layer3 = Dropout(0.2)(layer3)
    output = Dense(1, activation='sigmoid')(layer3)

    model = Model(input_layer, output)
    #loss_fn = tf.keras.losses.binary_crossentropy(from_logits=True)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    # original turbulence size
    nx, ny = 512, 512
    # down_sampling size
    ndx, ndy = 128, 128
    sample_size = 128 * 128
    # batch size
    batch_size = 100

    # load data
    phi1 = HMEq_data()
    ph_test = phi1.datafile_phi(ifrun=1)
    # create sampled batch
    ph_batch = ph_test.reshape((-1, nx, ny))
    ph_small_x = sampling_down(ph_batch).reshape(-1, sample_size)
    ph_small_y = np.ones(batch_size)
    # create random batch
    # x: input(2D image), y:output(0 for random, 1 for turbulence)
    rand_x, rand_y = gen_rand(batch_size, sample_size)
    X = np.append(ph_small_x, rand_x, axis=0)
    Y = np.append(ph_small_y, rand_y, axis=0)
    # shuffle datas
    for l in [X, Y]:
        np.random.seed(1)
        np.random.shuffle(l)

    # get a Keras calculation model
    model = full_connect(X[0])
    # split data into training and test data
    x_train, x_test = X[:100, :], X[100:, :]
    y_train, y_test = Y[:100], Y[100:]

#    model.fit(x_train, y_train, epochs=10)
# model.evaluate(x_test, y_test, verbose=2)
