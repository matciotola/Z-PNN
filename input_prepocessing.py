import math

import numpy as np
import torch
from skimage import transform
from torch import nn

import spectral_tools as ut
from interpolator_tools import interp23tap


def input_preparation(ms, pan, ratio, nbits, pad_size):
    """
        Prepare the remote sensing imagery for the pansharpening algorithm.
        In particular, the MS is upsampled with an ideal filter to the scale of PAN and a unique stack is created from
        both. After this, normalization is performed.

        Parameters
        ----------
        ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        ratio : int
            the resolution scale which elapses between MS and PAN.
        nbits : int
            the number of bits with which the images have been codified.
        pad_size : int
            Parameter linked to the scope of the network. It is used to perform the padding operation


        Return
        ------
        img_in : Numpy array
            the stack of MS + PAN images normalized as I = (I / 2 ** nbits)

        """

    ms = ms.astype(np.float64)
    pan = pan.astype(np.float64)
    max_value = 2 ** nbits
    ms = interp23tap(ms, ratio)

    pan = np.expand_dims(pan, axis=-1)

    img_in = np.concatenate([ms, pan], axis=-1) / max_value

    img_in = np.pad(img_in, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')

    return img_in


def resize_images(img_ms, img_pan, ratio, sensor=None, mtf=None, apply_mtf_to_pan=False):
    """
        Function to perform a downscale of all the data provided by the satellite.
        It downsamples the data of the scale value.
        To more detail please refers to

        [1] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods
        [2] L. Wald, (1) T. Ranchin, (2) M. Mangolini - Fusion of satellites of different spatial resolutions:
            assessing the quality of resulting images
        [3] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, M. Selva - MTF-tailored Multiscale Fusion of
            High-resolution MS and Pan Imagery
        [4] M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa - Pansharpening by convolutional neural networks in
            the full resolution framework


        Parameters
        ----------
        img_ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        img_pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        ratio : int
            the resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        mtf : Dictionary
            The desired Modulation Transfer Frequencies with which perform the low pass filtering process.
            Example of usage:
                MTF = {'GNyq' : np.asarray([0.21, 0.2, 0.3, 0.4]), 'GNyqPan': 0.5}
        apply_mtf_to_pan : bool
            Activate the downsample of the Panchromatic band with the Modulation Transfer Function protocol
            (Actually this feature is not used in our algorithm).


        Return
        ------
        I_MS_LR : Numpy array
            the stack of Multi-Spectral bands downgraded by the ratio factor
        I_PAN_LR : Numpy array
            The panchromatic band downsampled by the ratio factor

        """
    GNyq = []
    GNyqPan = []
    if (sensor is None) & (mtf is None):
        MS_scale = (math.floor(img_ms.shape[0] / ratio), math.floor(img_ms.shape[1] / ratio), img_ms.shape[2])
        PAN_scale = (math.floor(img_pan.shape[0] / ratio), math.floor(img_pan.shape[1] / ratio))
        I_MS_LR = transform.resize(img_ms, MS_scale, order=3)
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)

        return I_MS_LR, I_PAN_LR

    elif (sensor == 'QB') & (mtf is None):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
        GNyqPan = np.asarray([0.15])
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')) & (mtf is None):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
        GNyqPan = np.asarray([0.17])
    elif (sensor == 'GeoEye1' or sensor == 'GE1') & (mtf is None):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
        GNyqPan = np.asarray([0.16])
    elif (sensor == 'WV2') & (mtf is None):
        GNyq = 0.35 * np.ones((1, 7))
        GNyq = np.append(GNyq, 0.27)
        GNyqPan = np.asarray([0.11])
    elif (sensor == 'WV3') & (mtf is None):
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
        GNyqPan = np.asarray([0.5])
    elif mtf is not None:
        GNyq = mtf['GNyq']
        GNyqPan = np.asarray([mtf['GNyqPan']])

    N = 41

    b = img_ms.shape[-1]

    img_ms = np.moveaxis(img_ms, -1, 0)
    img_ms = np.expand_dims(img_ms, axis=0)

    h = ut.nyquist_filter_generator(GNyq, ratio, N)
    h = ut.mtf_kernel_to_torch(h)

    conv = nn.Conv2d(in_channels=b, out_channels=b, padding=math.ceil(N / 2),
                     kernel_size=h.shape, groups=b, bias=False, padding_mode='replicate')

    conv.weight.data = h
    conv.weight.requires_grad = False

    I_MS_LP = conv(torch.from_numpy(img_ms)).numpy()
    I_MS_LP = np.squeeze(I_MS_LP)
    I_MS_LP = np.moveaxis(I_MS_LP, 0, -1)
    MS_scale = (math.floor(I_MS_LP.shape[0] / ratio), math.floor(I_MS_LP.shape[1] / ratio), I_MS_LP.shape[2])
    PAN_scale = (math.floor(img_pan.shape[0] / ratio), math.floor(img_pan.shape[1] / ratio))

    I_MS_LR = transform.resize(I_MS_LP, MS_scale, order=0)

    if apply_mtf_to_pan:
        img_pan = np.expand_dims(img_pan, [0, 1])

        h = ut.nyquist_filter_generator(GNyqPan, ratio, N)
        h = ut.mtf_kernel_to_torch(h)

        conv = nn.Conv2d(in_channels=1, out_channels=1, padding=math.ceil(N / 2),
                         kernel_size=h.shape, groups=1, bias=False, padding_mode='replicate')

        conv.weight.data = h
        conv.weight.requires_grad = False

        I_PAN_LP = conv(torch.from_numpy(img_pan)).numpy()
        I_PAN_LP = np.squeeze(I_PAN_LP)
        I_PAN_LR = transform.resize(I_PAN_LP, PAN_scale, order=0)

    else:
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)

    return I_MS_LR, I_PAN_LR
