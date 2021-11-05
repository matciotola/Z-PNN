# Z-PNN: Zoom Pansharpening Neural Network
[Pansharpening by convolutional neural networks in the full resolution framework](https://www.tbd.com/) is 
a deep learning method for Pansharpening based on unsupervised and full-resolution framework training.

## Team members
 - Matteo Ciotola (matteo.ciotola@unina.it);
 - Sergio Vitale  (sergio.vitale@uniparthenope.it);
 - Antonio Mazza (antonio.mazza@unina.it);
 - Giovanni Poggi   (poggi@.unina.it);
 - Giuseppe Scarpa  (giscarpa@.unina.it).
 
 
## License
Copyright (c) 2021 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Prerequisites
All the functions and scripts were tested on Python 3.9, PyTorch 1.8.1 and 1.10.0, Cuda 10.1 and 11.3.
the operation is not guaranteed with other configurations.
The command to create the CONDA environment: 
```
conda env create -n z_pnn_env -f z_pnn_environment.yml
```

The command to activate the CONDA environment:
```
conda activate z_pnn_env
```


## Usage

### Before to start
The unique way to test this algorithm is through a `.mat` file. It must contain:
- `I_MS_LR`: Original Multi-Spectral Stack of dimensions in channel-last configurations (band index must be the last one);
- `I_PAN`: Original Panchromatic band, without the third dimension.

### Testing
The easiest command to use the algorithm on full resolution data:

```
python main.py -i path_to_mat_file -s sensor_name -o output_root  
```
Several options are possible. Please refer to the parser help for more details:

```
python main.py -h
```
