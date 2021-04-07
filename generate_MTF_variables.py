import numpy as np
import utils as ut
import scipy.ndimage.filters as ft
from xcorr import xcorr

def genMTF(ratio, sensor, N=41):
    if (sensor=='QB'):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22]) #Bands Order: B,G,R,NIR
    elif ((sensor=='Ikonos') or (sensor=='IKONOS')):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28]) #Bands Order: B,G,R,NIR
    elif (sensor=='GeoEye1'):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23]) #Bands Order: B, G, R, NIR
    elif (sensor=='WV2'):
        GNyq = 0.35* np.ones((1, 7)); GNyq = np.append(GNyq, 0.27)
    elif (sensor=='WV3'):
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]

    h = ut.NyquistFilterGenerator(GNyq,ratio, N)

    return h


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def generate_MTF_variables(ratio, sensor, PAN, MS):

    h = genMTF(ratio, sensor)

    nbands = h.shape[-1]

    P = np.round(ft.convolve(PAN, h[:,:,0], mode='nearest')).astype(np.uint16)

    rho = np.zeros((ratio, ratio, nbands))
    r = np.zeros((nbands,1)); np.squeeze(r)
    c = np.zeros((nbands,1)); np.squeeze(c)

    for i in range(int(ratio)):
        for j in range(int(ratio)):
            for b in range(nbands):
                temp = xcorr(MS[:,:,b], P[i::ratio, j::ratio], 2)
                rho[i,j,b] = np.mean(temp)
                temp = []

    for b in range(nbands):
        x = rho[:,:,b]
        max = x.max()
        pos = np.where(x == max)

        pos = tuple(map(int, pos))
        r[b] = pos[0]
        c[b] = pos[1]
        r = np.squeeze(r).astype(np.uint8)
        c = np.squeeze(c).astype(np.uint8)

    return h, r, c


if __name__ == '__main__':

    import scipy.io as io
    from interp23tap import interp23tap
    temp_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_1.mat'
    temp = io.loadmat(temp_path)
    I_PAN = temp['I_PAN'].astype('float64')
    I_MS = temp['I_MS_LR'].astype('float64')
    ratio = int(temp['ratio'][0][0])
    sensor = temp['sensor']
    aa = interp23tap(I_MS, 4)

    h, r, c = generate_MTF_variables(ratio, sensor, I_PAN, I_MS)