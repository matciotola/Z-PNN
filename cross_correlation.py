from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform.integral import integral_image as integral


def xcorr(img_1, img_2, half_width):
    """
        Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Numpy Array
            First image on which calculate the cross-correlation. Dimensions: H, W
        img_2 : Numpy Array
            Second image on which calculate the cross-correlation. Dimensions: H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation


        Return
        ------
        L : Numpy array
            The cross-correlation map between img_1 and img_2

    """
    w = ceil(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis=-1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis=-1)

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range(img_1.shape[-1]):
        img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
    for i in range(img_2.shape[-1]):
        img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

    img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w,
                                                                               :] + img_1_cum[:-2 * w, :-2 * w, :]) / (
                       4 * w ** 2)
    img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w,
                                                                               :] + img_2_cum[:-2 * w, :-2 * w, :]) / (
                       4 * w ** 2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu
    img_2 = img_2[w:-w, w:-w, :] - img_2_mu

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    i2 = img_1 ** 2
    j2 = img_2 ** 2
    ij = img_1 * img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

    sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :] + ij_cum[:-2 * w,
                                                                                                         :-2 * w, :])
    sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :] + i2_cum[:-2 * w,
                                                                                                         :-2 * w, :])
    sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :] + j2_cum[:-2 * w,
                                                                                                         :-2 * w, :])

    # sig2_ij_tot = np.clip(sig2_ij_tot, ep, sig2_ij_tot.max())
    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L


def xcorr_torch(img_1, img_2, half_width, device):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

    """
    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.type(torch.DoubleTensor)
    img_2 = img_2.type(torch.DoubleTensor)

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] - img_1_cum[:, :, 2 * w:, :-2 * w] +
                img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] - img_2_cum[:, :, 2 * w:, :-2 * w] +
                img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1 ** 2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2 ** 2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] - ij_cum[:, :, 2 * w:, :-2 * w] +
                   ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] - i2_cum[:, :, 2 * w:, :-2 * w] +
                   i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] - j2_cum[:, :, 2 * w:, :-2 * w] +
                   j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L
