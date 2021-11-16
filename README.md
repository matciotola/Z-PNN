# Z-PNN: Zoom Pansharpening Neural Network
[Pansharpening by convolutional neural networks in the full resolution framework](https://www.tbd.com/) is 
a deep learning method for Pansharpening based on unsupervised and full-resolution framework training.

## Team members
 - Matteo Ciotola (matteo.ciotola@unina.it);
 - Sergio Vitale  (sergio.vitale@uniparthenope.it);
 - Antonio Mazza (antonio.mazza@unina.it);
 - Giovanni Poggi   (poggi@unina.it);
 - Giuseppe Scarpa  (giscarpa@unina.it).
 
 
## License
Copyright (c) 2021 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/Z-PNN/LICENSE.txt)
(included in this package) 

## Prerequisites
All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

- Python 3.9 
- PyTorch 1.8.1 or 1.10.0
-  Cuda 10.1 and 11.3 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

- Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads) 
- Create a folder in which save the algorithm
- Download the algorithm and unzip it into the folder or, alternatively, from CLI:

```
git clone https://github.com/matciotola/Z-PNN
```

- Create the virtual environment with the `z_pnn_environment.yml`

```
conda env create -n z_pnn_env -f z_pnn_environment.yml
```

- Activate the Conda Environment

```
conda activate z_pnn_env
```

- Test it 

```
python main.py -i example/WV3_example.mat -o ./Output_Example -s WV3 -m Z-PNN --coregistration --view_results 
```


## Usage

### Before to start
To test this algorithm it is needed to create a `.mat` file. It must contain:
- `I_MS_LR`: Original Multi-Spectral Stack in channel-last configuration (Dimensions: H x W x B);
- `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: H x W).

### Testing
The easiest command to use the algorithm on full resolution data:

```
python main.py -i path/to/file.mat -s sensor_name -m method
```
Several options are possible. Please refer to the parser help for more details:

```
python main.py -h
```
