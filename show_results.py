from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from interp23tap import interp23tap


def view(I_MS, I_PAN, out, ratio, net_scope):
    Q_MS = np.quantile(I_MS, (0.02, 0.98), (0, 1), keepdims=True)
    Q_PAN = np.quantile(I_PAN, (0.02, 0.98), (0, 1), keepdims=True)

    ms_shape = (I_MS.shape[0] * ratio, I_MS.shape[1] * ratio, I_MS.shape[2])

    I_MS_LR_4x = resize(I_MS, ms_shape, order=0)[net_scope:-net_scope, net_scope:-net_scope, :]
    I_interp = interp23tap(I_MS, ratio)[net_scope:-net_scope, net_scope:-net_scope, :]

    DP = out - I_interp
    Q_d = np.quantile(abs(DP), 0.98, (0, 1))

    RGB = (4, 2, 1)
    RYB = (4, 3, 1)

    plt.figure()
    ax1 = plt.subplot(2, 4, 1)
    plt.imshow(I_PAN - Q_PAN[0, :, :] / (Q_PAN[1, :, :] - Q_PAN[0, :, :]), cmap='gray')
    ax1.set_title('PAN')

    T = (I_MS_LR_4x - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax2 = plt.subplot(2, 4, 2, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax2.set_title('MS (RGB)')

    ax6 = plt.subplot(2, 4, 6, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax6.set_title('MS (RYB)')

    T = (out - Q_MS[0, :, :]) / (Q_MS[1, :, :] - Q_MS[0, :, :])
    T = np.clip(T, 0, 1)

    ax3 = plt.subplot(2, 4, 3, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax3.set_title('ZPNN (RGB)')

    ax7 = plt.subplot(2, 4, 7, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax7.set_title('ZPNN (RYB)')

    T = 0.5 + DP / (2 * Q_d)
    T = np.clip(T, 0, 1)

    ax4 = plt.subplot(2, 4, 4, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RGB])
    ax4.set_title('Detail (RGB)')

    ax8 = plt.subplot(2, 4, 8, sharex=ax1, sharey=ax1)
    plt.imshow(T[:, :, RYB])
    ax8.set_title('Detail (RYB)')

    return
