import numpy as np
import torch
import math
"""
def fspecial_gauss(size, sigma):
    #"Function to mimic the 'fspecial' gaussian MATLAB function
    "
    x, y = np.mgrid[-size/2 + 1:size/2 + 1, -size/2 + 1:size/2 + 1]
    g = (np.exp(-(( x**2 + y** 2) / (2.0 * sigma ** 2)))).astype(np.float64)
    return g / g.sum()
"""

def fspecial_gauss(size, sigma):
    # Function to mimic the 'fspecial' gaussian MATLAB function

    m,n = [(ss-1.)/2. for ss in size]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    #h = np.round(h, 4)
    return h

def fir_filter_wind(Hd, w):
    """
    compute fir filter with window method
    Hd:     desired freqeuncy response (2D)
    w:      window (2D)
    """

    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)

    return h


def NyquistFilterGenerator(Gnyq, ratio, N):

    assert isinstance(Gnyq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(Gnyq, list):
        Gnyq = np.asarray(Gnyq)

    nbands = Gnyq.shape[0]


    kernel = np.zeros((N, N, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(Gnyq[j])))
        H = fspecial_gauss((N,N), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(N, 0.5)

        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))
    #kernel = np.round(kernel, 4)
    return kernel


## To eval
def net_scope(kernel_size):
    scope = 0
    for i in range(len(kernel_size)):
        scope += math.floor(kernel_size[i]/2)
    return scope
