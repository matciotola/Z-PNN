import math
from math import floor

import numpy as np
import torch
import torch.nn as nn

from cross_correlation import xcorr_torch
from spectral_tools import gen_mtf


def net_scope(kernel_size):
    """
        Compute the network scope.

        Parameters
        ----------
        kernel_size : List[int]
            A list containing the kernel size of each layer of the network.

        Return
        ------
        scope : int
            The scope of the network

        """

    scope = np.sum([math.floor(k / 2) for k in kernel_size])
    return scope


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False

    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask
