import argparse
import gc
import os
import shutil
from tqdm import tqdm

import numpy as np
import scipy.io as io
import torch
import torch.optim as optim

import networks
import losses
import utils
from input_prepocessing import input_preparation, resize_images
from sensor import Sensor
from spectral_tools import generate_mtf_variables
from show_results import view


def main_zpnn(args):
    # Parameter definitions

    test_path = args.input
    sensor = args.sensor
    method = args.method
    out_dir = args.out_dir
    epochs = args.epochs

    if epochs == 1 and 'Z-' in method:
        epochs = 100
    elif epochs == 1 and 'FT-Z' in method:
        epochs = 2000

    gpu_number = str(args.gpu_number)
    use_cpu = args.use_cpu
    reduce_res_flag = args.RR
    coregistration_flag = args.coregistration
    save_losses_trend_flag = args.save_loss_trend
    view_results_flag = args.view_results
    save_weights_flag = args.save_weights

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

    # Hyperparameters definition
    semi_width = 8

    # Torch configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")

    # Load test images
    temp = io.loadmat(test_path)

    I_PAN = temp['I_PAN'].astype('float32')
    I_MS = temp['I_MS_LR'].astype('float32')

    # class "Sensor" definition and PNN network definition
    s = Sensor(sensor)

    if 'DRPNN' in method:
        net = networks.DRPNN(s.nbands + 1)
    elif 'PanNet' in method:
        net = networks.PanNet(s.nbands, s.ratio)
    else:
        net = networks.PNN(s.nbands + 1, s.kernels, s.net_scope)

    if args.learning_rate != -1.0:
        s.learning_rate = args.learning_rate
    if args.beta != -1.0:
        s.beta = args.beta

    # Wald's Protocol
    if reduce_res_flag:
        I_MS, I_PAN = resize_images(I_MS, I_PAN, s.ratio, s.sensor)

    # Input preparation
    I_in = input_preparation(I_MS, I_PAN, s.ratio, s.nbits, s.net_scope)

    # Images reshaping for PyTorch workflow
    I_in = np.moveaxis(I_in, -1, 0)
    I_in = np.expand_dims(I_in, axis=0)
    I_inp = np.copy(I_in)
    I_in = I_in[:, :, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope]

    I_in = torch.from_numpy(I_in).float()
    I_inp = torch.from_numpy(I_inp).float()

    threshold = utils.local_corr_mask(I_in, s.ratio, s.sensor, device, semi_width)
    threshold = threshold.float()

    if (method == 'Z-PNN') or (method == 'A-PNN-FT-Z'):
        spec_ref = I_in[:, :-1, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope]
        struct_ref = torch.unsqueeze(I_in[:, -1, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope], dim=1)
        threshold = threshold[:, :, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope]
    else:
        spec_ref = I_in[:, :-1, :, :]
        struct_ref = torch.unsqueeze(I_in[:, -1, :, :], dim=1)
        s.net_scope = 0
        I_inp = I_in

    # Loading of pre-trained weights

    weight_path = 'weights/' + s.sensor + '_' + method + '_model.tar'
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
    else:
        print('Weights are not available for this method. Training from scratch will be performed.')

    # Losses definition
    if coregistration_flag:
        LSpec = losses.SpectralLoss(generate_mtf_variables(s.ratio, sensor, I_PAN, I_MS),
                                    s.net_scope,
                                    I_PAN.shape,
                                    s.ratio,
                                    device)
    else:
        LSpec = losses.SpectralLossNocorr(generate_mtf_variables(s.ratio, sensor, I_PAN, I_MS),
                                          s.net_scope,
                                          I_PAN.shape,
                                          s.ratio,
                                          device)

    LStruct = losses.StructuralLoss(s.ratio, device)

    # Fitting strategy definition
    net = net.to(device)
    if 'DRPNN' in method:
        params = []
        base_params = []
        for i, k in net.named_parameters():
            if i == 'Conv_11.weight':
                params.append(k)
            elif i == 'Conv_11.bias':
                params.append(k)
            else:
                base_params.append(k)

        optimizer = optim.Adam([
            {"params": base_params}, {"params": params, "lr": s.learning_rate * 1e-1}],
            lr=s.learning_rate)
    else:
        optimizer = optim.Adam(net.parameters(), lr=s.learning_rate)

    net.train()

    # Moving everything on the device
    I_in = I_in.to(device)
    spec_ref = spec_ref.to(device)
    struct_ref = struct_ref.to(device)
    threshold = threshold.to(device)

    LSpec = LSpec.to(device)
    LSpec.mask = LSpec.mask.to(device)
    LStruct = LStruct.to(device)

    # Best model path implementation
    temp_path = 'temp/'
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    path_min_loss = temp_path + 'weights_' + test_path.split(os.sep)[-1].split('.')[0] + '.tar'

    # Training
    history_loss = np.zeros(epochs)
    history_loss_spec = np.zeros(epochs)
    history_loss_struct = np.zeros(epochs)

    min_loss = np.inf

    pbar = tqdm(range(epochs), dynamic_ncols=True, initial=1)

    for epoch in pbar:

        running_loss = 0.0
        running_spec_loss = 0.0
        running_struct_loss = 0.0

        for i in range(I_in.shape[0]):
            inputs = I_in[i, :, :, :].view([1, I_in.size()[1], I_in.size()[2], I_in.size()[3]])
            labels_spec = spec_ref[i, :, :, :].view([1, spec_ref.size()[1], spec_ref.size()[2], spec_ref.size()[3]])
            labels_struct = struct_ref[i, :, :, :].view(
                [1, struct_ref.size()[1], struct_ref.size()[2], struct_ref.size()[3]])

            optimizer.zero_grad()
            outputs = net(inputs)
            loss_spec = LSpec(outputs, labels_spec)
            loss_struct, loss_struct_no_threshold = LStruct(outputs, labels_struct, threshold)
            loss = loss_spec + s.beta * loss_struct
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_spec_loss += loss_spec.item()
            running_struct_loss += loss_struct_no_threshold

        if running_loss < min_loss:
            min_loss = running_loss
            torch.save(net.state_dict(), path_min_loss)
        history_loss[epoch] = running_loss
        history_loss_spec[epoch] = running_spec_loss
        history_loss_struct[epoch] = running_struct_loss
        pbar.set_postfix(
            {'Overall Loss': running_loss, 'Spectral Loss': running_spec_loss, 'Structural Loss': running_struct_loss})

    # Output Folder creation

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Testing
    I_inp = I_inp.to(device)
    net.load_state_dict(torch.load(path_min_loss))
    net.eval()
    outputs = net(I_inp)

    out = outputs.cpu().detach().numpy()
    out = np.squeeze(out)
    out = np.moveaxis(out, 0, -1)
    out = out * (2 ** s.nbits)
    out = np.clip(out, 0, out.max())

    out = out.astype(np.uint16)
    save_path = out_dir + test_path.split(os.sep)[-1].split('.')[0] + '_' + method + '.mat'
    io.savemat(save_path, {'I_MS': out})

    if save_losses_trend_flag:
        io.savemat(
            out_dir + test_path.split(os.sep)[-1].split('.')[0] + '_losses_trend.mat',
            {
                'overall_loss': history_loss,
                'spectral_loss': history_loss_spec,
                'strucutral_loss: ': history_loss_struct
            }
        )

    if not view_results_flag:
        view(I_MS, I_PAN, out, s.ratio, method)

    torch.cuda.empty_cache()
    gc.collect()
    if save_weights_flag:
        shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Z-PNN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Z-PNN is a deep learning algorithm for remote sensing '
                                                 'imagery which performs pansharpening.',
                                     epilog='''\
Reference: 
Pansharpening by convolutional neural networks in the full resolution framework
M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa 
                                
Authors: 
Image Processing Research Group of University Federico II of Naples 
('GRIP-UNINA')
                                     '''
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-i", "--input", type=str, required=True,
                               help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')

    requiredNamed.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1'],
                               help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1)')

    requiredNamed.add_argument('-m', '--method', type=str, required=True, choices=["A-PNN-FT-Z", "Z-PNN",
                                                                                   "PanNet-FT-Z", "Z-PanNet",
                                                                                   "Z-DRPNN"],
                               default="Z-PNN", help='The algorithm with which perform Pansharpening.')

    default_out_path = 'Outputs/'
    optional.add_argument("-o", "--out_dir", type=str, default=default_out_path,
                          help='The directory in which save the outcome.')
    optional.add_argument("--epochs", type=int, default=1, help='Number of the epochs with which perform the '
                                                                'fine-tuning of the algorithm.')
    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')
    optional.add_argument("--RR", action="store_true", help='For evaluation only. The algorithm '
                                                            'will be performed at reduced '
                                                            'resolution.')
    optional.add_argument("--coregistration", action="store_true", help="Enable the co-registration feature.")
    optional.add_argument("--save_loss_trend", action="store_true", help="Option to save the trend of losses "
                                                                         "(For Debugging Purpose).")
    optional.add_argument("--view_results", action="store_true", help="Enable the visualization of the outcomes.")
    optional.add_argument("--save-weights", action="store_true", help="Save the training weights.")
    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')
    optional.add_argument("-b", "--beta", type=float, default=-1.0,
                          help='Beta value with which to weight the structural loss during the training.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main_zpnn(arguments)
