import torch
import numpy as np
import utils as ut
import torch.nn as nn
import math
import torchvision
from skimage import transform


def resize_images(I_MS, I_PAN, ratio, sensor=None, MTF=None):

    if (sensor==None) & (MTF==None):
        MS_scale = (math.floor(I_MS.shape[0] / ratio), math.floor(I_MS.shape[1] / ratio), I_MS.shape[2])
        PAN_scale = (math.floor(I_PAN.shape[0] / ratio), math.floor(I_PAN.shape[1] / ratio))
        I_MS_LR = transform.resize(I_MS, MS_scale, order=3)
        I_PAN_LR = transform.resize(I_PAN, PAN_scale, order=3)

        return I_MS_LR, I_PAN_LR

    elif (sensor=='QB') & (MTF==None):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22]) #Bands Order: B,G,R,NIR
        GNyqPan = 0.15
    elif ((sensor=='Ikonos') or (sensor=='IKONOS')) & (MTF==None):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28]) #Bands Order: B,G,R,NIR
        GNyqPan = 0.17
    elif (sensor=='GeoEye1') & (MTF==None):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23]) #Bands Order: B, G, R, NIR
        GNyqPan = 0.16
    elif (sensor=='WV2') & (MTF==None):
        GNyq = 0.35* np.ones((1, 7)); GNyq = np.append(GNyq, 0.27)
        GNyqPan = 0.11
    elif (sensor=='WV3') & (MTF==None):
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
        GNyqPan = 0.5
    elif (MTF!=None):
        GNyq = MTF['GNyq']
        GNyqPan = MTF['GNyqPan']

    N = 41

    r, c, b = I_MS.shape

    I_MS = np.moveaxis(I_MS,-1,0)
    I_MS = np.expand_dims(I_MS, axis=0)

    h = ut.NyquistFilterGenerator(GNyq,ratio, N)
    h = np.moveaxis(h, -1,0)
    h = np.expand_dims(h, axis=(1))
    h = h.astype('float32')

    h = torch.from_numpy(h)

    conv = nn.Conv2d(in_channels=b, out_channels=b, padding=math.ceil(N/2),
                     kernel_size=h.shape, groups=b, bias=False, padding_mode='replicate')

    conv.weight.data = h
    conv.weight.requires_grad = False

    I_MS_LP = conv(torch.from_numpy(I_MS)).numpy()
    I_MS_LP = np.squeeze(I_MS_LP); I_MS_LP = np.moveaxis(I_MS_LP, 0, -1)
    MS_scale = (math.floor(I_MS_LP.shape[0]/ratio), math.floor(I_MS_LP.shape[1]/ratio), I_MS_LP.shape[2])
    PAN_scale = (math.floor(I_PAN.shape[0]/ratio), math.floor(I_PAN.shape[1]/ratio))

    I_MS_LR = transform.resize(I_MS_LP, MS_scale, order=0)

    I_PAN_LR = transform.resize(I_PAN, PAN_scale, order=3)


    return I_MS_LR, I_PAN_LR



if __name__ == '__main__':
    import scipy.io as io

    temp_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_1.mat'
    temp = io.loadmat(temp_path)
    I_PAN = temp['I_PAN'].astype('float32')
    I_MS = temp['I_MS_LR'].astype('float32')
    ratio = int(temp['ratio'][0][0])
    sensor = temp['sensor']

    A, B = resize_images(I_MS, I_PAN, ratio, sensor)
