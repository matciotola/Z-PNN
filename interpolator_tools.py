import math
import numpy as np
import scipy.ndimage.filters as ft
import torch
import torch.nn as nn


def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img


def interp23tap_torch(img, ratio, device):
    """
        A PyTorch implementation of the Polynomial interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))
    BaseCoeff = np.concatenate([BaseCoeff] * b, axis=0)

    BaseCoeff = torch.from_numpy(BaseCoeff).to(device)
    img = img.astype(np.float32)
    img = np.moveaxis(img, -1, 0)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros((b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c))

        if z == 0:
            I1LRU[:, 1::2, 1::2] = img
        else:
            I1LRU[:, ::2, ::2] = img

        I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(torch.from_numpy(I1LRU).to(device), 2, 3))
        img = conv(torch.transpose(t, 2, 3)).cpu().detach().numpy()
        img = np.squeeze(img)

    img = np.moveaxis(img, 0, -1)

    return img
