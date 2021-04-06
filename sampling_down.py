import numpy as np


def sampling_down(phi, ndx=128, ndy=128):
    nx = phi.shape[0]
    ny = phi.shape[1]
    xstep = nx//ndx
    ystep = ny//ndy
    phi_down = phi[::xstep, ::ystep]
    return phi_down


ph_npz = np.load('diff-2force1_res512psi10_ph.npz')
# 最初のほうのデータは過渡応答っぽいので除外する
ph_data = ph_npz["ph"][400:]

ph_LR = np.zeros((1, 128, 128), dtype=float)
ph_HR = np.zeros((1, 512, 512), dtype=float)

for orig in ph_data:
    orig_max = orig.max()
    orig = orig/orig_max
    ph_HR = np.vstack((ph_HR, [orig]))
    tmp = sampling_down(orig)
    ph_LR = np.vstack((ph_LR, [tmp]))

print(ph_LR.shape)
print(ph_HR.shape)
np.savez('diff-2force1_psi10_HRLR.npz', ph_HR=ph_HR, ph_LR=ph_LR)
