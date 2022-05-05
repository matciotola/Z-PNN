import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

from interpolator_tools import interp23tap


def show(starting_img_ms, img_pan, algorithm_outcome, ratio, method, q_min=0.02, q_max=0.98):
    """
        Auxiliary function for results visualization.

        Parameters
        ----------
        starting_img_ms : Numpy Array
            The Multi-Spectral image. Dimensions: H, W, Bands
        img_pan : Numpy Array
            The PAN image. Dimensions: H, W
        algorithm_outcome : NumPy Array
            The Fused image. Dimensions: H, W, Bands
        ratio : int
            PAN-MS resolution ratio
        method : str
            The name of the pansharpening algorithm
        q_min : float
            Minimum quantile to compute, which must be between 0 and 1 inclusive.
        q_max : float
            Maximum quantile to compute, which must be between 0 and 1 inclusive.

        Return
        ------
        None

    """

    Q_MS = np.quantile(starting_img_ms, (q_min, q_max), (0, 1), keepdims=True)
    Q_PAN = np.quantile(img_pan, (q_min, q_max), (0, 1), keepdims=True)

    ms_shape = (starting_img_ms.shape[0] * ratio, starting_img_ms.shape[1] * ratio, starting_img_ms.shape[2])

    I_MS_LR_4x = resize(starting_img_ms, ms_shape, order=0)
    I_interp = interp23tap(starting_img_ms, ratio)

    DP = algorithm_outcome - I_interp
    Q_d = np.quantile(abs(DP), q_max, (0, 1))
    if starting_img_ms.shape[-1] == 8:
        RGB = (4, 2, 1)
        RYB = (4, 3, 1)
    else:
        RGB = (2, 1, 0)
        RYB = (2, 3, 0)
    plt.figure()
    ax1 = plt.subplot(2, 4, 1)
    plt.imshow((img_pan - Q_PAN[0, :, :]) / (Q_PAN[1, :, :] - Q_PAN[0, :, :]), cmap='gray')
    ax1.set_title('PAN')

    T = (I_MS_LR_4x - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax2 = plt.subplot(2, 4, 2, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax2.set_title('MS (RGB)')

    ax6 = plt.subplot(2, 4, 6, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax6.set_title('MS (RYB)')

    T = (algorithm_outcome - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax3 = plt.subplot(2, 4, 3, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax3.set_title(method + ' (RGB)')

    ax7 = plt.subplot(2, 4, 7, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax7.set_title(method + ' (RYB)')

    T = 0.5 + DP / (2 * Q_d)
    T = np.clip(T, 0, 1)

    ax4 = plt.subplot(2, 4, 4, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax4.set_title('Detail (RGB)')

    ax8 = plt.subplot(2, 4, 8, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax8.set_title('Detail (RYB)')
    plt.show()
    return
