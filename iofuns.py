# Copyright (c) 2020,
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratoire de Traitement des Signaux 4 (LTS4).
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Mattia Rossi (rossi-mattia-at-gmail-com)

import numpy as np
from typing import Tuple, Dict
import os
import re
import sys
import struct
from colmap.read_model import read_cameras_binary
from cv2 import imread


def read_depth_map(path: str, data_format: str,
                   size: Tuple[int, int] = None, stereo_param: Dict = None) -> np.array:
    """It reads a depth map either in the ETH3D format or in the COLMAP one.

    Args:
        path: path to the depth map.
        data_format: depth map format (`ETH3D`, `COLMAP`, `MIDDLEBURY`).
        size: a 2-Tuples specifying the depth map height and width (mandatory only for the ETH3D format).
        stereo_param: stereo parameters (mandatory for `data_format` equal to `'MIDDLEBURY'`).

    Returns:
        The read depth map (in meters) arranged as an `(H, W)` array. Non valid values are signalled with zero entries.
    """

    # Read the depth map.
    if data_format == 'ETH3D':

        # Depth map dimensions.
        if size is not None:
            height, width = size
        else:
            raise ValueError('For ETH3D depth type, the `size` parameter is mandatory.')

        with open(path, "rb") as fid:
            depth = np.reshape(np.fromfile(fid, dtype=np.float32), (height, width))
            # Note that depth values are of type np.float32.

    elif data_format == 'COLMAP':

        depth = read_bin_file(path)

    elif data_format == 'MIDDLEBURY':

        assert stereo_param is not None, 'For `data_format` equal to MIDDLEBURY, `stereo_param` is mandatory.'

        # Read the disparity map.
        disparity = read_middlebury_disparity(path)

        # Convert the read disparity to depth.
        depth = (stereo_param['baseline'] * stereo_param['cam0'][0, 0]) / (disparity - stereo_param['doffs']) / 1000.0

    else:

        raise ValueError('Bad depth format.')

    # Depending on the source, non valid pixels are signalled either with non positive entries or with infinite ones.

    # Signal the non valid entries with zero.
    depth[(depth < 0) | (depth == float('inf'))] = 0

    return depth


def read_normal_map(path: str, data_format: str) -> np.array:
    """It reads a normal map in the COLMAP format.

    Args:
        path: path to the normal map.
        data_format: the normal map format (currently, only 'COLMAP').

    Returns:
        The read normal map arranged as an `(H, W, 3)` array.
    """

    if data_format == 'COLMAP':

        normal = read_bin_file(path)

    else:

        raise ValueError('Bad normal format.')

    return normal


def read_confidence_map(path: str):

    confidence = (imread(path)[:, :, 0]).astype(np.float64) / 255.0

    return confidence


def read_middlebury_disparity(file_name: str) -> np.array:
    """It read the Middlebury 2014 dataset disparity.

    Args:
        file_name: path to the disparity map.

    Returns:
        The loaded disparity map, arranged as an `(H, W)` array.
    """

    # Read the disparity.
    disparity = load_pfm(file_name)

    # It is necessary to flip the disparity upside down.
    disparity = np.flipud(disparity)

    # Non valid disparity entries are signalled with infinite.

    return disparity


def read_middlebury_calib_file(file_name: str) -> Tuple:
    """It reads the calibration file of a scene from the Middlebury 2014 training dataset.

    It reads the calibration file of a scene from the Middlebury 2014 training dataset provided in
    `http://vision.middlebury.edu/stereo/data/scenes2014/`

    Args:
        file_name: calibration file name.

    Returns:
        A dictionary containing all the calibration file parameters.
        - cam0: left camera intrinsic matrix, arranged as a `(3, 3)` array,
        - cam1: camera intrinsic matrix, arranged as a `(3, 3)` array,
        - doffs: correction offset,
        - baseline: camera baseline,
        - width: image width,
        - height: image height,
        - ndisp: ground truth disparity resolution,
        - isint: ... ,
        - vmin: minimum disparity,
        - vmax: maximum disparity,
        - dyavg: ... ,
        - dymax: ... .
    """

    # Create the parameter dictionary.
    param = {}

    with open(file_name) as fp:

        # Read the left camera intrinsic matrix.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        data = data.replace('[', '').replace(']', '').replace(';', '')
        cam0 = np.reshape(np.fromstring(data, dtype=np.float32, sep=' '), (3, 3))
        param['cam0'] = cam0

        # Read the right camera intrinsic matrix.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        data = data.replace('[', '').replace(']', '').replace(';', '')
        cam1 = np.reshape(np.fromstring(data, dtype=np.float, sep=' '), (3, 3))
        param['cam1'] = cam1

        # Read doffs.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        doffs = float(data)
        param['doffs'] = doffs

        # Read baseline.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        baseline = float(data)
        param['baseline'] = baseline

        # Read width.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        width = int(data)
        param['width'] = width

        # Read height.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        height = int(data)
        param['height'] = height

        # Read ndisp.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        ndisp = int(data)
        param['ndisp'] = ndisp

        # Read isint.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        isint = int(data)
        param['isint'] = isint

        # Read vmin.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        vmin = float(data)
        param['vmin'] = vmin

        # Read vmax.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        vmax = float(data)
        param['vmax'] = vmax

        # Read dyavg.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        dyavg = float(data)
        param['dyavg'] = dyavg

        # Read dymax.
        line = fp.readline()
        data = (line.split('='))[1].rstrip()
        dymax = float(data)
        param['dymax'] = dymax

    return param


def read_kitti_calib_file(filename: str) -> np.array:
    """It reads the calibration file of a scene from the KITTI 2015 training dataset.

    It reads the calibration file of a scene from the KITTI 2015 training dataset provided in
    `http://www.cvlibs.net/datasets/kitti/`.

    Args:
        filename: calibration file name.

    Returns:
        A dictionary containing all the calibration file parameters.
        - P_rect_02: rectified left color camera intrinsic matrix, arranged as a `(3, 3)` array,
        - P_rect_03: rectified right color camera intrinsic matrix, arranged as a `(3, 3)` array,
        - baseline: rectified camera baseline.
    """

    param = {}

    with open(filename) as fp:

        while True:

            # Read a new line.
            line = fp.readline()

            # If the line is empty, the end of the file has been reached.
            if line == '':
                break

            # Split the line into parameter name and it value.
            param_name, data = line.split(':', maxsplit=1)

            # If the current line contains one of the parameters of interest, save it.
            if param_name == 'P_rect_02':
                param[param_name] = np.reshape(
                    np.fromstring(data.lstrip().rstrip(), dtype=np.float32, sep=' '),
                    (3, 4))
            elif param_name == 'T_02':
                param[param_name] = np.reshape(
                    np.fromstring(data.lstrip().rstrip(), dtype=np.float32, sep=' '),
                    (3,))
            elif param_name == 'P_rect_03':
                param[param_name] = np.reshape(
                    np.fromstring(data.lstrip().rstrip(), dtype=np.float32, sep=' '),
                    (3, 4))
            elif param_name == 'T_03':
                param[param_name] = np.reshape(
                    np.fromstring(data.lstrip().rstrip(), dtype=np.float32, sep=' '),
                    (3,))

    # Check that all the parameters have been read.
    assert 'P_rect_02' in param, 'Could not read left camera intrinsic matrix.'
    assert 'T_02' in param, 'Could not read left camera translation vector.'
    assert 'P_rect_03' in param, 'Could not read right camera intrinsic matrix.'
    assert 'T_03' in param, 'Could not read right camera translation vector.'

    # Compute the baseline.
    param['baseline'] = abs(
        (param['P_rect_02'][0, 3] / param['P_rect_02'][0, 0]) - (param['P_rect_03'][0, 3] / param['P_rect_03'][0, 0]))

    return param


def read_bin_file(file_name: str) -> np.array:
    """It reads a depth map or normal map in the COLMAP bin format.

    It reads a depth map or normal map in the COLMAP bin format. In practice, it can read any 2D or 3D array.
    It is a modified version of COLMAP `read_array()` Python script.

    Args:
        file_name: source file.

    Returns:
        A depth map or a normal map, arranged as an `(H, W)` or an `(H, W, 3)` array, respectively.
    """

    with open(file_name, "rb") as fid:

        # Read the file header. It is in the format 'width&height&channel_nb&' where channel_nb is 1 for 2D data.
        width, height, channel_nb = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)

        while True:

            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break

            byte = fid.read(1)

        # Read the data, stored as float32 in C-like order.
        data = np.fromfile(fid, np.float32)

    # Reshape the read data into a (width, height, channel_nb) array.
    data = data.reshape((width, height, channel_nb), order='F')

    # Transpose the data to get an array in the (height, width, channel_nb) format.
    data = np.transpose(data, (1, 0, 2))

    # In the case of 2D data, remove the last dimension.
    if channel_nb == 1:
        data = data[:, :, 0]

    return data


def write_bin_file(data: np.array, file_name: str) -> None:
    """It writes a depth map or a normal map in the COLMAP bin format.

    It writes a depth map or a normal map in the COLMAP bin format. In practice, it can write any 2D or 3D array.

    Args:
        data: depth map or normal map, arranged as an `(H, W)` or an `(H, W, 3)` array, respectively.
        file_name: destination file name.
    """

    # Check the input data.
    assert data.ndim == 2 or data.ndim == 3, 'The input data must be 2D or 3D.'

    # If the input data are 2D, a fake 3D dimension is added. This permits to treat 2D and 3D data the same way.
    if data.ndim == 2:
        data = data[:, :, None]

    # Number of color channels.
    channel_nb = data.shape[2]

    with open(file_name, "wb") as file:

        # Write the file header.
        file.write(bytearray(str(data.shape[1]), 'utf8'))
        file.write(bytearray('&', 'utf8'))
        file.write(bytearray(str(data.shape[0]), 'utf8'))
        file.write(bytearray('&', 'utf8'))
        file.write(bytearray(str(channel_nb), 'utf8'))
        file.write(bytearray('&', 'utf8'))

    # Write the data.
    with open(file_name, "ab") as file:

        for c in range(channel_nb):
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    file.write(struct.pack('f', data[y, x, c]))


def load_pfm(file_name: str) -> np.array:
    """It reads a pfm file.

    It reads a pfm file. Adapted from the following web page:
    `https://stackoverflow.com/questions/48809433/read-pfm-format-in-python`

    Args:
        file_name: PFM file name.

    Returns:
        The PFM file, arranged as an `(H, W, C)` or an `(H, W)` array.
    """

    with open(file_name, "rb") as f:

        # Line 1: the number of channels.
        channel_type = f.readline().decode('latin-1')
        if "PF" in channel_type:
            channels = 3

        elif "Pf" in channel_type:
            channels = 1

        else:
            sys.exit(1)

        # Line 2: height and width.
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: positive number means big endian, negative means little endian.
        line = f.readline().decode('latin-1')
        big_endian = True
        if "-" in line:
            big_endian = False

        # Slurp all the binary data.
        samples = width * height * channels;
        buffer = f.read(samples * 4)

        # Unpack the floats with the appropriate endianness.
        if big_endian:
            fmt = ">"
        else:
            fmt = "<"

        fmt = fmt + str(samples) + "f"
        data = struct.unpack(fmt, buffer)

        # Reshape the data.
        data = np.reshape(np.array(data), (height, width, channels)).squeeze()

    return data


def colmap_camera_intrinsic(path: str) -> Dict[str, float]:
    """It reads the camera intrinsic parameters stored in COLMAP format.

    It reads the intrinsic parameters of the first pinhole camera stored in a `.txt` or `.bin` COLMAP camera file.

    Args:
        path: path to the COLMAP file.

    Returns:
        A dictionary with the following fields:
        - the horizontal and vertical focal lengths `f_x` and `f_y`,
        - the horizontal and vertical coordinates of the camera center of projection `c_x` and `c_y`.
    """

    camera = read_cameras_binary(path)[0]

    assert camera.model == 'PINHOLE', 'The input camera must refer to a pinhole model.'

    focal_x, focal_y, center_x, center_y = camera.params

    camera_intrinsic = dict(
        f_x=focal_x,
        f_y=focal_y,
        c_x=center_x,
        c_y=center_y)

    return camera_intrinsic


def read_camera_data_text(path: str) -> Dict:
    """It extracts the camera info stored in a text file (Andreas' format).

    Args:
        path: path to the camera info file.

    Returns:
        A dictionary containing all the camera info.
    """

    # Dictionary containing the camera data.
    camera = {
        'height': None,
        'width': None,
        'A': None,
        'k1': None,
        'k2': None,
        'R': None,
        'T': None,
        'zmin': None,
        'zmax': None,
        'match': None}

    # Extract the camera data from the text file.
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                line_elems = line.split('=')
                if len(line_elems) != 2:
                    continue
                param_name = line_elems[0]
                param_value = line_elems[1]
                param_name_elems = param_name.split('.')
                if (len(param_name_elems) != 2) or (param_name_elems[0] != 'camera'):
                    continue
                param_name = param_name_elems[1]
                camera[param_name] = param_value

    # Convert the camera height to `int`.
    if (camera['height'] is not None) and ('.' not in camera['height']):
        camera['height'] = int(camera['height'])
    else:
        camera['height'] = None

    # Convert the camera width to `int`.
    if (camera['width'] is not None) and ('.' not in camera['width']):
        camera['width'] = int(camera['width'])
    else:
        camera['width'] = None

    # Convert the camera `k1' parameter to `float`.
    if camera['k1'] is not None:
        camera['k1'] = int(camera['k1'])
    else:
        camera['k1'] = None

    # Convert the camera `k2' parameter to `float`.
    if camera['k2'] is not None:
        camera['k2'] = int(camera['k2'])
    else:
        camera['k2'] = None

    # Convert the camera intrinsic matrix to `np.array`.
    if camera['A'] is not None:
        mtx_intrinsic = [float(i) for i in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', camera['A'])]
        mtx_intrinsic = np.asarray(mtx_intrinsic, dtype=float)
        if len(mtx_intrinsic) != 9:
            camera['A'] = None
        else:
            camera['A'] = np.reshape(mtx_intrinsic, (3, 3))

    # Convert the camera rotation matrix to `np.array`.
    if camera['R'] is not None:
        mtx_rotation = [float(i) for i in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', camera['R'])]
        mtx_rotation = np.asarray(mtx_rotation, dtype=float)
        if len(mtx_rotation) != 9:
            camera['R'] = None
        else:
            camera['R'] = np.reshape(mtx_rotation, (3, 3))

    # Convert the camera translation vector to `np.array`.
    if camera['T'] is not None:
        vec_translation = [float(i) for i in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', camera['T'])]
        vec_translation = np.asarray(vec_translation, dtype=float)
        if len(vec_translation) != 3:
            camera['T'] = None
        else:
            camera['T'] = np.reshape(vec_translation, (3, 1))

    return camera
