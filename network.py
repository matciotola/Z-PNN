import torch.nn as nn
import torch.nn.functional as F
import torch
from generate_MTF_variables import generate_MTF_variables
from math import floor, ceil
import numpy as np
from utils import net_scope
from xcorr import xcorr_torch as xcorr

class PNN(nn.Module):
    def __init__(self, in_channels, kernels, scope):
        super(PNN, self).__init__()

        ##Network variables
        self.scope = scope

        ##Network structure
        self.conv1 = nn.Conv2d(in_channels, 48, kernels[0])
        self.conv2 = nn.Conv2d(48, 32, kernels[1])
        self.conv3 = nn.Conv2d(32, in_channels - 1, kernels[2])

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + inp[:, :-1, self.scope:-self.scope, self.scope:-self.scope]
        return x


class SpectralLoss(nn.Module):
    def __init__(self, ratio, sensor, PAN, MS, device, mask=None):

        # Class initialization
        super(SpectralLoss, self).__init__()

        # Parameters definition
        self.sensor = sensor
        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.nbands = 4
        elif (sensor == 'WV2') or (sensor == 'WV3'):
            self.nbands = 8

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'WV2') or (sensor == 'WV3'):
            kernels = [9, 5, 5]
        elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            kernels = [5, 5, 5]
        self.net_scope = net_scope(kernels)

        self.ratio = int(ratio)
        self.device = device


        # Conversion of filters in Tensor
        MTF_kern, self.MTF_r, self.MTF_c = generate_MTF_variables(ratio, sensor, PAN, MS)
        self.pad = floor((MTF_kern.shape[0] - 1) / 2)

        MTF_kern = np.moveaxis(MTF_kern, -1, 0)
        MTF_kern = np.expand_dims(MTF_kern, axis = 1)

        self.MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)


        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=self.MTF_kern.shape,
                                   bias=False)

        self.depthconv.weight.data = self.MTF_kern
        self.depthconv.weight.requires_grad = False

        # Mask definition
        if mask != None:
            self.mask = mask
        else:
            self.mask = torch.ones((1, self.nbands, PAN.shape[-2]-(self.net_scope+self.pad)*2, PAN.shape[-1]-(self.net_scope+self.pad)*2))

    def forward(self, outputs, labels):

        x = self.depthconv(outputs)
        ## TO DO: assign a dynamic range to valid region
        labels = labels[:,:, 20:-20,20:-20]
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)



        for b in range(self.nbands):
            y[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]

        W_ = W_ / torch.sum(W_)

        L = torch.sum((x - y)**2 * W_, dim=(-2, -1))
        L = torch.mean(L)

        return L


class StructuralLoss(nn.Module):

    def __init__(self, sigma, beta, xcorr_th, gains):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma/2)
        self.beta = beta
        self.xcorr_th = xcorr_th
        self.gains = gains


    def forward(self, outputs, labels):

        X = -torch.ones(outputs.shape)

        X = torch.max(X, xcorr(outputs, labels, self.scale))
        X = torch.ones(X.shape) - X

        #Lxcorr_no_weights = torch.mean(X, dim=(-1, -2))
        worst = X.gt(self.xcorr_th)


        X = X*(~worst)*self.gains[0] + X*(worst)*self.gains[1]
        Lxcorr = torch.mean(X)


        return self.beta * Lxcorr



if __name__ == '__main__':

    """
    net = PNN(9, 9)
    print(net)

    inp = torch.ones((1, 9, 256, 256))
    output = torch.zeros((1, 8, 240, 240))

    out = net(inp)
    """

    import scipy.io as io
    from interp23tap import interp23tap

    temp_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_1.mat'
    temp = io.loadmat(temp_path)
    I_PAN = temp['I_PAN'].astype('float32')
    I_MS = temp['I_MS_LR'].astype('float32')
    ratio = int(temp['ratio'][0][0])
    sensor = temp['sensor']
    I_MS_FR = interp23tap(I_MS, 4)

    loss = SpectralLoss(ratio, sensor, I_PAN, I_MS)
    out = I_MS_FR[8:-8,8:-8,:]
    out = np.moveaxis(out, -1, 0)
    out = np.expand_dims(out, axis=0)
    out = torch.from_numpy(out).type(torch.float32)

    #ref = I_MS_FR[28:-28, 28:-28, :]
    ref = I_MS_FR[8:-8, 8:-8, :]
    ref = np.moveaxis(ref, -1, 0)
    ref = np.expand_dims(ref, axis=0)
    ref = torch.from_numpy(ref).float()

    x = loss(out, ref)

    sloss = StructuralLoss(ratio, 2e-4, 0.06, [1, 12])

    y = sloss(out, out[:,0,:,:])