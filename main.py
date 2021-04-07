import os
import scipy.io as io
import numpy as np
import torch
import torch.optim as optim

from interp23tap import interp23tap
from resize_images import resize_images
from sensor import Sensor
import network
from input_preparation import input_preparation
from matplotlib import pyplot as plt
from show_results import view

if __name__ == '__main__':

    ## Hyperparameters definition
    reduceResFlag = False

    epochs = 100
    beta = 2e-4
    spatial_gain = (1, 12)
    threshold = 0.06
    learning_rate = 1

    ## Test tile path
    test_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_4_zoom.mat'

    ## Torch configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load test images
    temp = io.loadmat(test_path)

    I_PAN = temp['I_PAN'].astype('float64')
    I_MS = temp['I_MS_LR'].astype('float64')
    sensor = temp['sensor'][0]
    ratio = int(temp['ratio'][0][0])

    ## Wald Protocol
    if reduceResFlag == True:
        I_MS_LR, I_PAN = resize_images()

    ## class "Sensor" definition -> PNN network definition
    s = Sensor(sensor)

    ## Load tuned network
    weight_path = 'weights/' + s.sensor + '_PNNplus_model.tar'
    s.net.load_state_dict(torch.load(weight_path))

    ## Losses definition
    LSpec = network.SpectralLoss(s.ratio, s.sensor, I_PAN, I_MS, device)
    LStruct = network.StructuralLoss(s.ratio, beta, threshold, spatial_gain)

    ## Input preparation

    I_in = input_preparation(I_MS, I_PAN, s.ratio, s.nbits, s.net_scope)

    ## Reshape of images for torch workflow

    cut_size = s.net_scope

    I_in = np.moveaxis(I_in, -1, 0)
    I_in = np.expand_dims(I_in, axis=0)
    I_inp = np.copy(I_in)
    I_in = I_in[:, :, cut_size:-cut_size, cut_size:-cut_size]

    cut_size = int(s.net_scope)

    I_in = torch.from_numpy(I_in).float()
    I_inp = torch.from_numpy(I_inp).float()

    spec_ref = I_in[:, :-1, cut_size:-cut_size, cut_size:-cut_size]
    struct_ref = torch.unsqueeze(I_in[:, -1, cut_size:-cut_size, cut_size:-cut_size], dim=0)

    ## Fitting strategy definition

    optimizer = optim.SGD(s.net.parameters(), lr=learning_rate, momentum=0.9)

    s.net = s.net.to(device)
    I_in = I_in.to(device)
    spec_ref = spec_ref.to(device)
    struct_ref = struct_ref.to(device)

    LSpec = LSpec.to(device)
    LSpec.mask = LSpec.mask.to(device)
    LStruct = LStruct.to(device)

    ## Best model path implementation

    min_loss = 1000

    temp_path = 'temp/'
    if os.path.exists(temp_path) == False:
        os.mkdir(temp_path)
    path_min_loss = temp_path + 'weights.tar'

    ## Training

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = s.net(I_in)
        loss_spec = LSpec(outputs, spec_ref)
        loss_struct = LStruct(outputs, struct_ref)
        loss = loss_spec + loss_struct
        loss.backward()
        optimizer.step()
        print('Epoch: %.3d, loss: %.5f' % (epoch + 1, loss))

        if loss_spec < min_loss:
            min_loss = loss_spec
            torch.save(s.net.state_dict(), path_min_loss)


    ## Testing
    I_inp = I_inp.to(device)
    s.net.load_state_dict(torch.load(path_min_loss))
    outputs = s.net(I_inp)

    RGB = (4, 2, 1)
    out = outputs.cpu().detach().numpy()
    out = np.squeeze(out)
    out = np.moveaxis(out, 0, -1)
    out = out * 2**s.nbits
    out = np.clip(out, 0, out.max())

    view(I_MS, I_PAN, out, s.ratio)

    out = out.astype(np.uint16)
    save_path = temp_path + test_path.split(os.sep)[-1].split('.')[0] + '_Z-PNN_wFT.mat'
    io.savemat(save_path, {'I_MS': out})


