import numpy as np
from skimage.util import pad
from interp23tap import interp23tap


def input_preparation(ms, pan, ratio, nbits, pad_size):

    ms = ms.astype(np.float64); pan = pan.astype(np.float64)
    max_value = 2**nbits
    ms = interp23tap(ms, ratio)

    pan = np.expand_dims(pan, axis=-1)

    img_in = np.concatenate([ms, pan], axis=-1)/max_value

    img_in = np.pad(img_in,((int(pad_size/2), int(pad_size/2)), (int(pad_size/2), int(pad_size/2)), (0,0)), 'edge')

    return img_in



if __name__ == '__main__':
    import scipy.io as io

    temp_path = '/home/matteo/Scrivania/PNN/Zoom_PNN_advanced_v0/Datasets-ZPNN/WV3_Adelaide_crops/Adelaide_1.mat'
    temp = io.loadmat(temp_path)
    I_PAN = temp['I_PAN'].astype('float64')
    I_MS1 = temp['I_MS_LR'].astype('float32')
    I_MS = temp['I_MS_LR'].astype('float64')
    ratio = int(temp['ratio'][0][0])
    sensor = temp['sensor']
    nbits = temp['L'][0][0]
    a = input_preparation(I_MS, I_PAN, ratio, nbits, 12)