from math import ceil
import numpy as np
from skimage.transform.integral import integral_image as integral
import torch
import torch.nn.functional as F


def xcorr(img_1, img_2, half_width):

    w = ceil(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis=-1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis=-1)

    img_1 = np.pad(img_1.astype(np.float64), ((w,w), (w,w), (0,0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w,w), (w,w), (0,0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range (img_1.shape[-1]):
        img_1_cum[:,:,i] = integral(img_1[:,:,i]).astype(np.float64)
    for i in range (img_2.shape[-1]):
        img_2_cum[:,:,i] = integral(img_2[:,:,i]).astype(np.float64)

    img_1_mu = (img_1_cum[2*w:,2*w:,:] - img_1_cum[:-2*w, 2*w:, :] - img_1_cum[2*w:, :-2*w, :] + img_1_cum[:-2*w, :-2*w,:]) / (4*(w)**2)
    img_2_mu = (img_2_cum[2*w:,2*w:,:] - img_2_cum[:-2*w, 2*w:, :] - img_2_cum[2*w:, :-2*w, :] + img_2_cum[:-2*w, :-2*w,:]) / (4*(w)**2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu; img_2 = img_2[w:-w, w:-w, :] - img_2_mu


    img_1 = np.pad(img_1.astype(np.float64), ((w,w), (w,w), (0,0))); img_2 = np.pad(img_2.astype(np.float64), ((w,w), (w,w), (0,0)))

    i2 = img_1**2
    j2 = img_2**2
    ij = img_1*img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:,:,i] = integral(i2[:,:,i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:,:,i] = integral(j2[:,:,i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:,:,i] = integral(ij[:,:,i]).astype(np.float64)


    sig2_ij_tot = (ij_cum[2*w:,2*w:,:] - ij_cum[:-2*w, 2*w:, :] - ij_cum[2*w:, :-2*w, :] + ij_cum[:-2*w, :-2*w,:])
    sig2_ii_tot = (i2_cum[2*w:,2*w:,:] - i2_cum[:-2*w, 2*w:, :] - i2_cum[2*w:, :-2*w, :] + i2_cum[:-2*w, :-2*w,:])
    sig2_jj_tot = (j2_cum[2*w:,2*w:,:] - j2_cum[:-2*w, 2*w:, :] - j2_cum[2*w:, :-2*w, :] + j2_cum[:-2*w, :-2*w,:])

    sig2_ij_tot = np.clip(sig2_ij_tot, ep, sig2_ij_tot.max())
    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    L = sig2_ij_tot / ((sig2_ii_tot*sig2_jj_tot)**0.5 + ep)

    return L


def xcorr_torch(img_1, img_2, half_width):

    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.type(torch.DoubleTensor); img_2 = img_2.type(torch.DoubleTensor)
    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim= -2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim= -2)

    img_1_mu = (img_1_cum[:, :, 2*w:,2*w:] - img_1_cum[:, :, :-2*w, 2*w:] - img_1_cum[:, :, 2*w:, :-2*w] + img_1_cum[:, :, :-2*w, :-2*w]) / (4*(w)**2)
    img_2_mu = (img_2_cum[:, :, 2*w:,2*w:] - img_2_cum[:, :, :-2*w, 2*w:] - img_2_cum[:, :, 2*w:, :-2*w] + img_2_cum[:, :, :-2*w, :-2*w]) / (4*(w)**2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu; img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1**2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2**2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1*img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2*w:,2*w:] - ij_cum[:, :, :-2*w, 2*w:] - ij_cum[:, :, 2*w:, :-2*w] + ij_cum[:, :, :-2*w, :-2*w])
    sig2_ii_tot = (i2_cum[:, :, 2*w:,2*w:] - i2_cum[:, :, :-2*w, 2*w:] - i2_cum[:, :, 2*w:, :-2*w] + i2_cum[:, :, :-2*w, :-2*w])
    sig2_jj_tot = (j2_cum[:, :, 2*w:,2*w:] - j2_cum[:, :, :-2*w, 2*w:] - j2_cum[:, :, 2*w:, :-2*w] + j2_cum[:, :, :-2*w, :-2*w])

    sig2_ij_tot = torch.clip(sig2_ij_tot, ep, sig2_ij_tot.max().item())
    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L


if __name__ == '__main__':

    import scipy.io as io
    from interp23tap import interp23tap
    temp_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_1.mat'
    temp = io.loadmat(temp_path)
    I_PAN = temp['I_PAN'].astype('float64')[:50,:50]
    I_MS = temp['I_MS_LR'].astype('float64')[:50,:50,0]
    MS = I_MS

    xc1 = xcorr(I_PAN, MS, 2)

    #MS = np.moveaxis(MS, -1, 0)
    MS = np.expand_dims(MS, axis=(0,1))
    PAN = np.expand_dims(I_PAN, axis=(0,1))

    MS = torch.from_numpy(MS)
    PAN = torch.from_numpy(PAN)
    xc2 = xcorr_torch(PAN, MS, 2)
    xcc2 = xc2.numpy()
    xcc2 = np.squeeze(xcc2, axis=0)
    xcc2 = np.moveaxis(xcc2, 0, -1)