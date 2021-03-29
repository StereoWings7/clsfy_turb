import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.fft import fftshift, ifft2


def fft_backward(pk, nx, ny):
    nkx = (nx-2)//3
    nky = (ny-2)//3
    pk_lx = 2*nkx+1
    pk_ly = nky+1

    pk_full = np.zeros((pk_lx, pk_ly+nky), dtype='c16')
    ph = np.zeros((nx, ny), dtype='c16')

    # in HMEq_FFTW, normalization coeff is applied at forward transformation
    # (caution!! ↑that treatment is different from ordinal FFT procedure!!# )
    # realistic condition enforcing pattern
    coeff_norm = nx*ny

    for i in range(pk_lx):
        for j in range(pk_ly):
            pk_full[i, j+nky] = coeff_norm*pk[i, j]
    for i in range(pk_lx):
        for j in range(1, nky+1):
            ikx = i-nkx
            iky = j
            pk_full[i, j] = coeff_norm*pk[nkx-ikx, nky-iky].conjugate()

    # copy from pk_full to ph
    for i in range(pk_lx):
        for j in range(pk_ly+nky):
            ist = nx//2-nkx
            jst = ny//2-nky
            ph[i+ist, j+jst] = pk_full[i, j]
    phr = fftshift(ph)
    return np.real(ifft2(phr))


def fftwavenumber_2d(lx, ly, ix, iy):
    dkx = np.pi/lx
    dky = np.pi/ly
    if(ix >= nkx):
        ikx = ix - nkx
    else:
        ikx = nkx-ix
    iky = iy
    kx = dkx*ikx
    ky = dky*iky
    return kx, ky


def out_enespect(lx, ly, nkx, nky, pk):
    nn = nky  # 1d energy spectrum DoF
    dk = np.pi/ly  # 1d energy spectrum wavenumber resolution
    enek = np.zeros(nn, dtype='float64')
    ekxy = np.zeros(((2*nkx+1), (nky+1)), dtype='float64')
    for i in range(2*nkx+1):
        for j in range(nky+1):
            kx, ky = fftwavenumber_2d(lx, ly, i, j)
            k2 = kx**2+ky**2
            kk = np.sqrt(k2)
            ekxy[i, j] = k2*np.abs(pk[i, j])**2
            for n in range(nn):
                if(kk >= dk*n and kk < dk*(n+1)):
                    enek[n] += ekxy[i, j]
    return ekxy, enek


def out_eneens(lx, ly, nkx, nky, ekxy):
    dkV = (np.pi/lx)*(np.pi/ly)
    ene = 0
    ens = 0
    for i in range(2*nkx+1):
        for j in range(nky+1):
            kx, ky = fftwavenumber_2d(lx, ly, i, j)
            k2 = kx**2+ky**2
            ene += dkV*ekxy[i, j]
            ens += dkV*k2*ekxy[i, j]
    return ene, ens


nx, ny = 512, 512
nkx, nky = np.int((nx-2)//3), np.int((ny-2)//3)
dkx, dky = np.pi/nkx, np.pi/nky
kx = np.linspace(-nkx*dkx, nkx*dkx, 2*nkx+1)
ky = np.linspace(0, nky*dky, nky+1)
KX, KY = np.meshgrid(kx, ky)
lx, ly = 300, 300
dx, dy = lx/nx, ly/ny
xx = np.linspace(0, lx, nx)
yy = np.linspace(0, ly, ny)
XX, YY = np.meshgrid(xx, yy)

with open("proc.dat") as fd_rstr:
    #print(type(fd_rstr))
    ifsta = int(fd_rstr.readline())
    ifend = int(fd_rstr.readline())
    print("start:{0},end:{1}".format(ifsta,ifend))


head = ("head", "<i")
tail = ("tail", "<i")
dt_pk = np.dtype([head, ("pk", "{0}<c16".format((2*nkx+1)*(nky+1))), tail])
dt_time = np.dtype([head, ("time", "<d"), tail])
dt_rstr = np.dtype(
    [("time", "<d"), ("time_out", "<d"), ("time_force", "<d"), ])
kk = np.arange(0, np.pi/ly*nky, np.pi/ly)
pk_size = dt_pk.itemsize
time_size = dt_time.itemsize
eneens_t = np.empty((0, 3), dtype=float)

# initial output
fd_pkinit = open("data/pk_001.bin", "rb")
chunk_pkinit = np.fromfile(fd_pkinit, dtype=dt_pk, count=2)
pk_init = chunk_pkinit[1]["pk"].reshape((2*nkx+1, nky+1), order="F")
ph_init = fft_backward(pk_init, nx, ny)
fig = plt.figure(figsize=(8.0, 6.0))
ax1 = fig.add_subplot(111)
mappable0 = ax1.pcolormesh(
    XX, YY, ph_init.T, shading='auto', cmap='jet')
pp = fig.colorbar(mappable0, ax=ax1, orientation='vertical')
pp.set_label("nagare func. amp.", fontsize=10)
plt.savefig(
    "pyfig/ph_000000.png")
plt.close()

# output binary file for later matplotlib
enek_iter = np.empty((0,nky), dtype=float)
ph_iter  = np.empty((0,nx,ny),dtype=float)
time_iter = np.empty((0,1),dtype=float)

iter_global = 0
output_time=10
for ifile in np.arange(ifsta,ifend+1):
    fd_pk = open("data/pk_{0:03}.bin".format(ifile), "rb")
    fd_time = open("data/time_{0:03}.bin".format(ifile), "rb")
    chunk_time = np.fromfile(fd_time, dtype=dt_time, count=10000)
    for iter in range(chunk_time.size):
    #for iter in range(1000):
        #print("time = {0}".format(iter_global))
        iter_global += 1
        if(iter_global % output_time == 0):
            print("output")
            fd_pk.seek(iter*pk_size)
            fd_time.seek(iter*time_size)
            chunk_pk = np.fromfile(fd_pk, dtype=dt_pk, count=1)
            chunk_time = np.fromfile(fd_time, dtype=dt_time, count=1)
            pk_tmp = chunk_pk[0]["pk"].reshape((2*nkx+1, nky+1), order="F")
            time_tmp = chunk_time[0]["time"]
            time_iter = np.append(time_iter,time_tmp)

            ekxy, enek = out_enespect(lx, ly, nkx, nky, pk_tmp)
            ene, ens = out_eneens(lx, ly, nkx, nky, ekxy)
            eneens_t = np.append(eneens_t, np.array(
                [[time_tmp, ene, ens]]), axis=0)
            ph_tmp = fft_backward(pk_tmp, nx, ny)

            # append for output
            ph_iter = np.append(ph_iter,np.array([ph_tmp]),axis=0)
            enek_iter = np.append(enek_iter,np.array([enek]),axis=0)

#            # visualization output
#            fig = plt.figure(figsize=(8.0, 6.0))
#            ax1 = fig.add_subplot(211)
#            ax2 = fig.add_subplot(212)
#            mappable0 = ax1.pcolormesh(KX, KY, ekxy.T, norm=colors.LogNorm(
#                vmin=np.min(ekxy[np.nonzero(ekxy)]), vmax=ekxy.max()), shading='auto', cmap='inferno')
#            pp = fig.colorbar(mappable0, ax=ax1, orientation='vertical')
#            pp.set_label("energy amp", fontsize=10)
#            ax2.plot(kk, enek)
#            ax2.set_xscale('log')
#            ax2.set_yscale('log')
#            plt.savefig(
#                "pyfig/enek_{0:06}.png".format(np.int(time_tmp//output_time)*output_time))
#                #"pyfig/enek_{0:07}.png".format(iter_global))
#            plt.close()
#            fig = plt.figure(figsize=(8.0, 6.0))
#            ax1 = fig.add_subplot(111)
#            mappable0 = ax1.pcolormesh(
#                XX, YY, ph_tmp.T, shading='auto', cmap='jet')
#            pp = fig.colorbar(mappable0, ax=ax1, orientation='vertical')
#            pp.set_label("nagare func. amp.", fontsize=10)
#            plt.savefig(
#                "pyfig/ph_{0:06}.png".format(np.int(time_tmp//output_time)*output_time))
#                #"pyfig/ph_{0:07}.png".format(iter_global))
#            plt.close()

#plt.plot(eneens_t[:, 0], eneens_t[:, 1], "r", label="energy")
#plt.plot(eneens_t[:, 0], eneens_t[:, 2], "k", label="enstrophy")
##plt.yscale("log")
#plt.legend()
#plt.savefig("pyfig/eneens.png")

# output npz files
np.savez('ph_enek_time',time=time_iter, ph=ph_iter,enek=enek_iter)
