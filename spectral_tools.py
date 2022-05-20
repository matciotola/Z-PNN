import numpy as np
import scipy.ndimage.filters as ft
import torch

from cross_correlation import xcorr


def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function

        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter

        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension

        """

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fir_filter_wind(hd, w):
    """
        Compute fir filter with window method

        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)

        Return
        ------
        h : Numpy array
            The fir Filter

    """

    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)

    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.

        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.

    """

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)

        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))

    return kernel


def gen_mtf(ratio, sensor, kernel_size=41):
    """
        Compute the estimated MTF filter kernels for the supported satellites.

        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.

        """
    GNyq = []

    if sensor == 'QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif (sensor == 'GeoEye1') or (sensor == 'GE1'):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif sensor == 'WV2':
        GNyq = 0.35 * np.ones((1, 7))
        GNyq = np.append(GNyq, 0.27)
    elif sensor == 'WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]

    h = nyquist_filter_generator(GNyq, ratio, kernel_size)

    return h


def generate_mtf_variables(ratio, sensor, img_pan, img_ms):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).

        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        img_pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        img_ms : Numpy Array
            Stack of Multi-Spectral bands. Dimension: H, W, B

        Return
        ------
        h : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.
        r : list
            The row misalignment biases
        c : list
            The column misalignment biases

        """

    h = gen_mtf(ratio, sensor)

    nbands = h.shape[-1]

    P = np.round(ft.convolve(img_pan, h[:, :, 0], mode='nearest')).astype(np.uint16)

    rho = np.zeros((ratio, ratio, nbands))
    r = np.zeros((nbands, 1))
    np.squeeze(r)
    c = np.zeros((nbands, 1))
    np.squeeze(c)

    for i in range(int(ratio)):
        for j in range(int(ratio)):
            for b in range(nbands):
                temp = xcorr(img_ms[:, :, b], P[i::ratio, j::ratio], 2)
                rho[i, j, b] = np.mean(temp)
                del temp

    for b in range(nbands):
        x = rho[:, :, b]
        max_value = x.max()
        pos = np.where(x == max_value)
        if len(pos[0]) != 1:
            pos = (pos[0][0], pos[1][0])
        pos = tuple(map(int, pos))
        r[b] = pos[0]
        c[b] = pos[1]
        r = np.squeeze(r).astype(np.uint8)
        c = np.squeeze(c).astype(np.uint8)

    return h, r, c


def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).

        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.

        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis=1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)
    return h
