import argparse

import numpy as np
import scipy.io as io
from osgeo import gdal


def tiff_to_mat_conversion(ms_path, pan_path, save_path, ms_initial_point=(0, 0), ms_final_point=(0, 0), ratio=4):
    """
        Generation of *.mat file, starting from the native GeoTiFF extension.
        Also, a crop tool is provided to analyze only small parts of the image.

        Parameters
        ----------
        ms_path : str
            The path of the Multi-Spectral image
        pan_path : str
            The path of the Panchromatic file
        save_path : str
            The destination mat file
        ms_initial_point : tuple
            Upper left point for image cropping. The point must be expressed in pixel coordinates,
            as (x,y), where (0,0) is precisely the point at the top left.
        ms_final_point : tuple
            Bottom right point for image cropping. The point must be expressed in pixel coordinates,
            as (x,y), where (0,0) is precisely the point at the top left.
        ratio : int
            The resolution scale which elapses between MS and PAN.

        Return
        ------
        I_in : Dictionary
            The dictionary, composed of MS and Pan images.

        """

    ms = gdal.Open(ms_path)
    ms = ms.ReadAsArray()
    ms = np.moveaxis(ms, 0, -1)

    pan = gdal.Open(pan_path)
    pan = pan.ReadAsArray()

    if ms_final_point[0] != 0 and ms_final_point[1] != 0:
        ms = ms[ms_initial_point[1]:ms_final_point[1], ms_initial_point[0]:ms_final_point[0], :]
        pan = pan[ms_initial_point[1] * ratio:ms_final_point[1] * ratio,
              ms_initial_point[0] * ratio:ms_final_point[0] * ratio]

    io.savemat(save_path, {'I_MS_LR': ms, 'I_PAN': pan})

    I_in = {'I_MS_LR': ms, 'I_PAN': pan}

    return I_in


def mat_to_tiff_conversion(mat_path, pan_path, save_path):
    """
        Conversion of the outcome of Full-Resolution framework algorithms from *.mat to GeoTiff.

        Parameters
        ----------
        mat_path : str
            The path of the outcome
        pan_path : str
            The path of the Panchromatic file
        save_path : str
            The destination mat file

        """

    ms = io.loadmat(mat_path)['I_MS'].astype(np.uint16)

    pan = gdal.Open(pan_path)

    geo_t = pan.GetGeoTransform()
    projection = pan.GetProjection()
    bands = ms.shape[-1]
    image_type = gdal.GDT_UInt16
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(save_path, ms.shape[1], ms.shape[0], bands, image_type)
    out.SetGeoTransform(geo_t)
    out.SetProjection(projection)

    for i in range(ms.shape[-1]):
        out.GetRasterBand(i + 1).WriteArray(ms[:, :, i])

    out = []

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TiffMatConversion',
                                     description='Script to convert GeoTiff file in *.mat for Z-PNN usage and the other way around.',
                                     )

    parser.add_argument('-m', '--mode', type=str, required=True, choices=["Tiff2Mat", "Mat2Tiff"],
                        default="Tif2Mat", help='The algorithm with which perform Pansharpening.')

    parser.add_argument('-ms', '--ms_tiff_path', type=str, help='The path of GeoTiff Multi-Spectral file.')
    parser.add_argument('-pan', '--pan_tiff_path', type=str, help='The path of GeoTiff Panchromatic file.')
    parser.add_argument('-mat', '--mat_path', type=str, help='The path of *.mat file - '
                                                             'Path of Full-Resolution framework result.')
    parser.add_argument('-o', '--out_path', type=str, help='The path where save the output.')
    parser.add_argument('--initial_point', nargs="+", type=int, default=[0, 0],
                        help='Upper left point for image cropping. The point must be expressed in pixel coordinates, '
                             'as x y, where 0 0 is precisely the point at the top left and referred to Multi-Spectral image.')
    parser.add_argument('--final_point', nargs="+", type=int, default=[0, 0],
                        help='Bottom right point for image cropping. The point must be expressed in pixel coordinates, '
                             'as x y, where 0 0 is precisely the point at the top left and referred to the Multi-Spectral image')

    arguments = parser.parse_args()

    initial_point = tuple(arguments.initial_point)
    final_point = tuple(arguments.final_point)

    if arguments.mode == 'Tiff2Mat':
        _ = tiff_to_mat_conversion(arguments.ms_tiff_path, arguments.pan_tiff_path, arguments.out_path,
                                   arguments.initial_point, arguments.final_point)
    elif arguments.mode == 'Mat2Tiff':
        mat_to_tiff_conversion(arguments.mat_path, arguments.pan_tiff_path, arguments.out_path)
    else:
        print('Unsupported choice.')
