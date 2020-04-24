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

import math
import numpy as np
import torch
from torch.nn import functional as fun
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from filters import gauss_filter_deriv_2d, diff_filter_bank
from transforms import depth2depth_inv
from typing import Tuple, Union


def resize_map(data: np.array, size_new: Tuple[int, int], order: int = 0) -> np.array:
    """It re-sizes the input map.

    It up-samples or down-samples any map (e.g., an image) with one or more channels.

    Args:
        data: map to resize, arranged as an `(H, W)` or `(H, W, C)` array.
        size_new: 2-tuple specifying the new height and width.
        order: order of the spline to be used in the re-sizing.

    Returns:
        The re-sized map, with dimensions `size_new[0], size_new[1]` or `size_new[0], size_new[1], C`. The output
        data type reflects the input one.
    """

    # Check that the data is either 2D or 3D.
    if (data.ndim != 2) & (data.ndim != 3):
        raise ValueError('Input data must be either 2D or 3D.')

    # Input data dimensions.
    height = data.shape[0]
    width = data.shape[1]

    # The target dimensions.
    height_new, width_new = size_new

    # We make the following assumptions:
    # - each pixel in the input data has height `1` and width `1`,
    # - `data[y, x]` is concentrated at the spatial coordinates `(y, x)`.
    # According to the previous two assumptions:
    # - the top left corner of the pixel associated to `data[y, x]` is at spatial coordinates `(y - 0.5, x - 0.5)`,
    # - the bottom right corner of the pixel associated to `data[y, x]` is at spatial coordinates `(y + 0.5, x + 0.5)`,
    # - `data` has its top left corner at the spatial coordinates `(- 0.5, - 0.5)`,
    # - `data` has its bottom right corner at the spatial coordinates `(height - 1 + 0.5, width - 1 + 0.5)`.

    # NOTE:
    # Re-sizing the input data means enlarging the pixel size, not decreasing the data (image or depth) area.
    # After resizing, the top left and bottom right corners of `data` will still be located at spatial coordinates
    # `(- 0.5, - 0.5)` and `(height - 1 + 0.5, width - 1 + 0.5)`, respectively.

    # New pixel dimensions.
    pixel_height_new = float(height) / height_new
    pixel_width_new = float(width) / width_new

    # Compute the coordinates of center of the top left pixel in the re-sized data.
    start_y = - 0.5 + (pixel_height_new / 2.0)
    start_x = - 0.5 + (pixel_width_new / 2.0)

    # Compute the coordinates of the center of the bottom right pixel in the new data.
    end_y = height - 1 + 0.5 - (pixel_height_new / 2.0)
    end_x = width - 1 + 0.5 - (pixel_width_new / 2.0)

    # Compute the new sampling grid.
    y_coord_new, x_coord_new = np.mgrid[start_y:end_y:(height_new * 1j), start_x:end_x:(width_new * 1j)]

    # Organize the sampling grid in a single array.
    points_new = np.stack((y_coord_new.flatten(), x_coord_new.flatten()), axis=1)

    # Re-sample the input depth.
    if data.ndim == 2:

        # Single channel input.

        aux = map_coordinates(data, points_new.T, order=order, mode='nearest')
        data_resized = np.reshape(aux, (height_new, width_new))

    else:

        # Multiple channel input.

        # Number of channels.
        channel_nb = data.shape[2]

        aux = tuple(
            map_coordinates(data[:, :, i], points_new.T, order=order, mode='nearest') for i in range(channel_nb))
        aux = np.stack(aux, axis=1)
        data_resized = np.reshape(aux, (height_new, width_new, channel_nb))

    return data_resized


def filler_1d(data: np.array, mask: np.array) -> np.array:
    """It fills a sparse 1D array.

    It fills the missing entries in the sparse 1D array `data` using linear interpolation.
    A missing entry can be filled if and only if it is included between two available entries.

    Args:
        data: sparse array of dimension `(N,)`.
        mask: array of dimension `(N,)` with `mask[i]` equal to `1` if the entry `data[i]` is available, equal to `0`
            if `data[i]` needs to be filled.

    Returns:
        A new filled array with `nan` values at those entries that could not be filled.
    """

    # Array length.
    length = len(data)

    # Array support.
    line = np.arange(0, length)

    # Compute the coordinates of the available entries.
    available_entries = line[mask.astype(np.bool)]

    # Compute the coordinates of the non available entries.
    target_entries = line[~mask.astype(np.bool)]

    # Allocate the filled array.
    data_filled = np.copy(data)

    # Perform the filling.
    if len(available_entries) <= 1:
        data_filled[~mask.astype(np.bool)] = math.nan
    else:
        data_filled[target_entries] = griddata(
            available_entries, (data[available_entries]), target_entries, method='linear')
    # The target entries at the left of the left-most available entry or at the right of the-right most available
    # entry are assigned the `nan` value.

    return data_filled


def filler_2d(data: np.array, mask: np.array) -> np.array:
    """It fills a sparse 2D array.

    It fills the missing entries in the sparse 2D array `data` using the following approach.
    First, two candidates are computed:
    - one obtained by interpolating linearly all the rows separately,
    - one obtained by interpolating linearly all the columns separately.
    Then, the derivative of the two candidates are computed:
    - the horizontal candidate derivative is obtained by deriving each row separately,
    - the vertical candidate derivative is obtained by deriving each column separately.
    For each missing entry, the candidate with the lowest derivative (in absolute value) is selected.

    Missing entries with only one candidate are assigned that candidate. Missing entries without any candidate are
    filled with nearest neighbor.

    Args:
        data: sparse array of dimensions `(H, W)`.
        mask: array of dimension `(H, W)` with `mask[i, j]` equal to `1` if the entry `data[i, j]` is available,
              equal to `0` if `data[i, j]` needs to be filled.

    Returns:
        A new filled array.
    """

    # Initialize the filled data with the input one.
    data_filled = np.copy(data)

    # Check whether there are entries to fill. If there are, then fill them.
    if np.sum(mask) != data.size:

        # Input data dimensions.
        height, width = data.shape

        # Perform the horizontal filling.
        data_horiz = np.zeros_like(data)
        for i in range(height):
            data_horiz[i, :] = filler_1d(data[i, :], mask[i, :])

        # Compute the horizontal derivative. `nan` derivatives are set to infinity.
        derivative_horiz = np.abs(np.diff(np.append(data_horiz, data_horiz[:, -2:-1], axis=1), axis=1))
        derivative_horiz[np.isnan(derivative_horiz)] = float('inf')

        # Perform the vertical filling.
        data_vert = np.zeros_like(data)
        for i in range(width):
            data_vert[:, i] = filler_1d(data[:, i], mask[:, i])

        # Compute the vertical derivative. `nan` derivatives are set to infinity.
        derivative_vert = np.abs(np.diff(np.append(data_vert, data_vert[-2:-1, :], axis=0), axis=0))
        derivative_vert[np.isnan(derivative_vert)] = float('inf')

        # Detect those pixels where the horizontal derivative is stronger than the vertical one, in absolute value.
        mask_orientation = derivative_horiz > derivative_vert

        # Perform the merging.
        data_filled = np.copy(data_horiz)
        data_filled[mask_orientation] = data_vert[mask_orientation]
        # Entries where no estimate is available (if any) are equal to `nan`.

        # Detect the entries where no estimate is available (if any), and fill them via nearest neighbor interpolation.
        mask_unfilled = np.isnan(data_filled)
        if np.sum(mask_unfilled) > 0:
            i, j = np.mgrid[0:data.shape[0]:1, 0:data.shape[1]:1]
            available_entries = np.stack((i[~mask_unfilled], j[~mask_unfilled]), axis=1)
            target_entries = np.stack((i[mask_unfilled], j[mask_unfilled]), axis=1)
            data_filled[mask_unfilled] = griddata(
                available_entries, (data_filled[~mask_unfilled]), target_entries, method='nearest')

    return data_filled


def filler_2d_nearest(data: np.array, mask: np.array) -> np.array:
    """It fills a sparse 2D array using nearest neighbour interpolation.

    Args:
        data: sparse array of dimensions `(H, W)`.
        mask: array of dimension `(H, W)` with `mask[i, j]` equal to `1` if the entry `data[i, j]` is available,
              equal to `0` if `data[i, j]` needs to be filled.

    Returns:
        A new filled array.
    """

    # Initialize the filled data with the input one.
    data_filled = np.copy(data)

    mask_available = mask.astype(np.bool)

    # Check whether there are entries to fill. If there are, then fill them.
    if np.sum(mask) != data.size:

        i, j = np.mgrid[0:data.shape[0]:1, 0:data.shape[1]:1]
        available_entries = np.stack((i[mask_available], j[mask_available]), axis=1)
        target_entries = np.stack((i[~mask_available], j[~mask_available]), axis=1)
        data_filled[~mask_available] = griddata(
            available_entries, (data_filled[mask_available]), target_entries, method='nearest')

    return data_filled


def similarity_graph(image: torch.Tensor,
                     window_size: int = 9, patch_size: int = 7,
                     sigma_intensity: float = 0.2, sigma_spatial: float = 3.0,
                     degree_max: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    """It builds a similarity graph on the input image.

    Args:
        image: reference image, arranged as a `(1, 1, H, W)` tensor.
        window_size: edge size of the square searching window.
        patch_size: edge size of the square patch used in the similarity computation.
        sigma_intensity: intensity standard deviation for the gaussian similarity weights.
        sigma_spatial: spatial standard deviation for the gaussian similarity weights.
        degree_max: maximum number of neighbors for each node (pixel) in the similarity graph.

    Returns:
        A tuple containing two `(1, degree_max, H, W)` tensors. The entry `(0, k, i, j)` of the first tensor stores the
        similarity weight between the pixels `(i, j)' of the input image and its k-th best neighbor.
        The linear index of k-th best neighbor is stored in the entry `(0, k, i, j)` of the second tensor.
        A pixel `(i, j)` with less than `degree_max` neighbors has the array `(0, :, i, j)` in the first tensor filled
        with zeros. The linear index, in the second tensor, associated to the aforementioned zero weights is the linear
        index of the pixel `(i, j)` itself.
    """

    # Check the input image type.
    assert image.is_floating_point(), "The input image must be of type float."

    # Image dimensions.
    channel_nb = image.size(1)
    height = image.size(2)
    width = image.size(3)

    # Organize the channels in the batch dimension.
    image_aux = image
    if channel_nb > 1:
        image_aux = image.transpose(0, 1).contiguous()

    # Create the filters to be used to compute the patch similarity.
    filter_bank = diff_filter_bank(window_size).to(image_aux)

    # Compute the padding for the patch similarity computation.
    window_radius = int((window_size - 1) / 2.0)
    patch_radius = int((patch_size - 1) / 2.0)
    pad = [window_radius + patch_radius] * 4

    # Compute the pixel similarity.
    pixel_similarity = fun.conv2d(
        fun.pad(image_aux, pad, mode='replicate'), filter_bank).pow(2).sum(dim=0, keepdim=True)
    # `pixel_similarity` is `(1, window_size * window_size, height + (2 * patch_radius), width + (2 * patch_radius))`.

    # Compute the integral image associated to `similarity`.
    pad = (1, 0, 1, 0)      # (pad_left, pad_right, pad_top, pad_bottom)
    integral = fun.pad(pixel_similarity, pad, mode='constant', value=0).cumsum(dim=2).cumsum(dim=3)
    # `integral` is `(1, window_size * window_size, height + (2 * patch_radius) + 1, width + (2 * patch_radius) + 1)`.

    # Free the memory associated to `pixel_similarity`.
    del pixel_similarity

    # Exploit the integral image to compute the patch similarity in constant time.
    integral_height = integral.size(2)
    integral_width = integral.size(3)
    bottom_right = integral.narrow(2, integral_height - height, height).narrow(3, integral_width - width, width)
    bottom_left = integral.narrow(2, integral_height - height, height).narrow(3, 0, width)
    top_right = integral.narrow(2, 0, height).narrow(3, integral_width - width, width)
    top_left = integral.narrow(2, 0, height).narrow(3, 0, width)
    patch_similarity = bottom_right.clone().add_(-1.0, bottom_left).add_(-1.0, top_right).add_(top_left)

    # DEBUG.
    # patch_similarity.sqrt_()

    # Normalize the patch similarity.
    patch_similarity.div_((- 2.0) * (sigma_intensity ** 2))

    # Free the memory associated to `integral`.
    del integral

    # Define the window grid.
    y_window, x_window = torch.meshgrid(
        [torch.arange(- window_radius, window_radius + 1, dtype=torch.int16, device=image_aux.device),
         torch.arange(- window_radius, window_radius + 1, dtype=torch.int16, device=image_aux.device)])
    y_window = y_window.reshape(1, -1)
    x_window = x_window.reshape(1, -1)

    # Remove the entry `(0, 0)` from the window grid, as `filter_bank` does not contain any filter for this coordinate.
    mask = (y_window == 0) & (x_window == 0)
    y_window = y_window[~mask].reshape(1, -1, 1, 1)
    x_window = x_window[~mask].reshape(1, -1, 1, 1)

    # Compute the squared spatial distance.
    spatial_weights = x_window.to(patch_similarity).pow_(2) + y_window.to(patch_similarity).pow_(2)

    # Normalize the spatial distance.
    spatial_weights.div_((- 2.0) * (sigma_spatial ** 2))

    # Compute the global weights (based on both patch similarity and spatial distance).
    weights = patch_similarity.add_(spatial_weights).exp_()
    # weights = patch_similarity.exp_()       # DEBUG.

    # Define the image grid.
    y_source, x_source = torch.meshgrid(
        [torch.arange(height, dtype=torch.int16, device=image_aux.device),
         torch.arange(width, dtype=torch.int16, device=image_aux.device)])
    y_source = y_source[None, None,]
    x_source = x_source[None, None,]

    # Detect and remove the non valid weights, i.e., those associated to pixel outside the actual image support.
    y_target = torch.zeros_like(y_source)
    x_target = torch.zeros_like(x_source)
    for i in range(weights.size(1)):

        # Compute the neighbouring pixel coordinates.
        torch.add(y_source, y_window.narrow(1, i, 1), out=y_target)
        torch.add(x_source, x_window.narrow(1, i, 1), out=x_target)

        # Detect the non valid coordinates and set them to zero.
        weights.narrow(1, i, 1).mul_(
            (y_target >= 0).to(weights)).mul_(
            (y_target < height).to(weights)).mul_(
            (x_target >= 0).to(weights)).mul_(
            (x_target < width).to(weights))

    # For each pixel, select the `degree_max` neighbours with the largest weights.
    weights_top, indexes = torch.topk(weights, degree_max, dim=1)
    # Note that, although the weights associated to non valid neighbours have been set equal to zero, some of these
    # neighbours may still have been selected. This must be taken into account later.

    # Free the memory associated to `weights`.
    del weights

    # Normalize the vector of weights associated to each pixel by its sum.
    weights_top.div_(
        torch.max(weights_top.sum(dim=1, keepdim=True).expand_as(weights_top), weights_top.new_ones(1) * 1e-12))

    # Build the tensor `indexes_linear`.
    index_linear = torch.zeros_like(weights_top, dtype=torch.long)
    for i in range(degree_max):

        # Flatten the spatial dimensions of `indexes`.
        indexes_flattened = indexes.narrow(1, i, 1).view(1, -1, 1, 1)

        # Compute the neighboring pixel coordinates.
        torch.add(
            y_source,
            torch.gather(y_window, 1, indexes_flattened).view(y_source.size()),
            out=y_target)
        torch.add(
            x_source,
            torch.gather(x_window, 1, indexes_flattened).view(x_source.size()),
            out=x_target)

        # The coordinates of the non valid neighbors of a pixel `p` are set equal to the coordinates of `p` itself.
        mask = None
        if (y_target < 0).any() or (y_target >= height).any():
            mask = (y_target < 0) | (y_target >= height)
            y_target[mask] = y_source[mask]
        if (x_target < 0).any() or (x_target >= width).any():
            mask = (x_target < 0) | (x_target >= width)
            x_target[mask] = x_source[mask]

        # Convert the spatial indexes into linear.
        torch.add(
            x_target.to(index_linear),
            width,
            y_target.to(index_linear),
            out=index_linear.narrow(1, i, 1))

    # Free the memory associated to `y_target`, `x_target`, `mask`.
    del y_target, x_target, mask

    return weights_top, index_linear


def unravel_index(index: Union[np.ndarray, torch.Tensor], size: Tuple[int, int])\
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """It converts linear indexes into matrix indexes.

    It converts each input linear index `i` into a pair `(row, col)` for the matrix whose shape is specified at the input.

    Args:
        index: linear indexes, arranged as an `(N,)` array or tensor.
        size: matrix shape.

    Returns:
        A tuple containing the row and columns indexes, each one arranged as an `(N,)` array or tensor.
    """

    height, width = size

    # # Check the class of the input data.
    # index_class = type(index).__name__
    #
    # if index_class == 'ndarray':
    #
    #     row, col = np.divmod(index, width)
    #
    # elif index_class == 'Tensor':
    #
    #     row = (index.div(width)).floor_()
    #     col = index.fmod(width)
    #
    # else:
    #
    #     raise TypeError('The input index data type must be ndarray or Tensor.')

    row = index // width
    col = index % width

    return row, col


def depth_percentage_error(depth: np.array, depth_gt: np.array, threshold: float):
    """It computes the percentage of pixel whose depth has an error larger than a predefined threshold.

    Args:
        depth: depth map to check, arranged as an `(H, W)` array.
        depth_gt: ground truth depth map, arranged as an `(H, W)` array.
        threshold: error threshold.

    Returns:
        The percentage of pixels in the input depth map with an error larger than the specified threshold.
    """

    mask = (depth_gt > 0) & (depth_gt < float('inf'))
    error = np.abs(depth_gt - depth)
    error = (np.sum(error[mask] > threshold) / np.sum(mask)) * 100

    return error


def space2plane_normal(depth: np.array, normal: np.array,
                       focal: Tuple[float, float], center: Tuple[float, float]) -> np.array:
    """It computes the 2D normals associated to the inverse depth, starting from the 3D normals.

    The unitary normal associated to a 3D points `(X_0, Y_0, Z_0)` defines a plane `P` that locally approximates the
    surface around the point itself. Let us indicate with `(x_0, y_0)` the coordinates of the projection of
    `(X_0, Y_0, Z_0)` onto the camera image plane. Assuming a pinhole camera model, the inverse depth associated to
    the plane `P` is a plane as well, `P1` hereafter, passing through the point `(x_0, y_0, 1 / depth[x_0, x_0])`.
    In particular, the plane `P1` is described by the following equation:

                    `(1 / depth[x, y]) = (1 / depth[x_0, y_0])  +  (w_1 * (x - x_0))  +  (w_2 * (y - y_0))`

    where the direction of the (non necessarily unitary) vector `(w_0, w_1, -1)` defines the orientation of `P1`.
    For each pixel in the input depth map, this function leverages the normal of the corresponding 3D point to compute
    the corresponding vector `(w_0, w_1)`.

    Input 3D normals with the 'z' component equal to zero are mapped to the 2D zero vector.
    Input 3D normals whose corresponding depth is not valid are mapped to the 2D zero vector.

    Args:
        depth: depth map, arranged as an `(H, W)` array.
        normal: normal map, arranged as an `(H, W, 3)` array. Normals must be unitary.
        focal: tuple containing the (pinhole) camera focal length along the `x` and 'y' axes, respectively.
        center: `x` and `y` coordinates of the center of (pinhole) camera center of projection.

    Returns:
        The 2D normals associated to the input 3D normals, arranged as an `(H, W, 2)` array.
    """

    # Define the data type to be used below: 64-bit precision is recommended.
    dtype = np.float64

    # Convert the input depth map to `dtype`.
    d = depth.astype(dtype, copy=False)

    # Depth map dimensions.
    height = depth.shape[0]
    width = depth.shape[1]

    # Build the depth map grid.
    x, y = np.meshgrid(np.arange(width, dtype=dtype), np.arange(height, dtype=dtype))

    # Extract the camera focal lengths and the coordinates of the camera center of projection.
    focal_x, focal_y = focal
    center_x, center_y = center

    # Detect the entries of the grid where the depth is available.
    mask = (d > 0) & (d < float('inf'))

    # Create a copy of the 3D normals where those associated to non available depth entries are set to zero.
    normal_new = np.zeros_like(normal, dtype=dtype)
    normal_new[mask] = normal[mask]

    # Re-normalize the normals.
    normal_norm = np.linalg.norm(normal_new, axis=2)
    mask_nnz = (normal_norm > 0)
    for i in range(3):
        normal_new[:, :, i][mask_nnz] = normal_new[:, :, i][mask_nnz] / normal_norm[mask_nnz]

    # Name the 3D normal components as in the report.
    a = normal_new[:, :, 0]
    b = normal_new[:, :, 1]
    c = normal_new[:, :, 2]

    # Compute the cosine of the angle between the 3D normal and the line of sight of the corresponding 3D point.
    rho = np.zeros_like(depth, dtype=dtype)
    rho[mask] = d[mask] * (
            ((a[mask] * (x[mask] - center_x)) / focal_x) +
            ((b[mask] * (y[mask] - center_y)) / focal_y) +
            c[mask])

    '''
    # Cases:
    # 1. A 3D normal with negative `rho` indicates a 3D point on the side of a plane visible by the camera.
    # 2. A 3D normal with positive `rho` indicates a 3D point on the side of a plane hidden to the camera.
    #    However, it is sufficient to flip the normal orientation in order to associate the point to the side of the
    #    plane visible by the camera.
    # 3. A 3D normal with zero `rho` indicates a 3D point on a plane aligned with the line of sight of the point and
    #    therefore not visible by the camera (regardless of the side of the plane).
    #
    # As the normals `n` and `-n` are both projected to the same 2D vector, it is not necessary to flip the normals
    # corresponding to the Case 2.
    '''

    # Allocate the space for the 2D normals and name them as in the report.
    plane_normal = np.zeros((height, width, 2), dtype=dtype)
    w_0 = plane_normal[:, :, 0]
    w_1 = plane_normal[:, :, 1]

    # Compute the 2D normals associated to the available 3D normals.
    mask = (mask & (rho != 0))
    w_0[mask] = a[mask] / (rho[mask] * focal_x)
    w_1[mask] = b[mask] / (rho[mask] * focal_y)

    # The 3D normals with a valid depth, but corresponding to the Case 3 (i.e., `rho == 0`), are not valid.
    # These 3D normals are arbitrarily mapped to the 2D normal `[0, 0]`.

    return plane_normal


def plane2space_normal(depth: np.array, normal: np.array,
                       focal: Tuple[float, float], center: Tuple[float, float]) -> np.array:
    """It reverts the operation performed by `space2plane_normals`.

    Args:
        depth: depth map, arranged as an `(H, W)` array.
        normal: normal map, arranged as an `(H, W, 2)` array.
        focal: tuple containing the (pinhole) camera focal length along the `x` and 'y' axes, respectively.
        center: `x` and `y` coordinates of the center of (pinhole) camera center of projection.

    Returns:
        The 3D normals associated to the input 2D normals, arranged as an `(H, W, 3)` array.
    """

    # Define the data type to be used below: 64-bit precision is recommended.
    dtype = np.float64

    # Convert the input depth map to `dtype`.
    d = depth.astype(dtype, copy=False)

    # Depth map dimensions.
    height = depth.shape[0]
    width = depth.shape[1]

    # Build the depth map grid.
    x, y = np.meshgrid(np.arange(width, dtype=dtype), np.arange(height,dtype=dtype))

    # Extract the camera focal lengths and the coordinates of the camera center of projection.
    focal_x, focal_y = focal
    center_x, center_y = center

    # Detect the entries of the grid where the depth is available.
    mask = (d > 0) & (d < float('inf'))

    # Address the 2D normals as in the report.
    w_0 = normal[:, :, 0].astype(dtype, copy=False)
    w_1 = normal[:, :, 1].astype(dtype, copy=False)

    # Compute the coefficients of the first linear equation.
    alpha = np.zeros_like(depth, dtype=dtype)
    beta = np.zeros_like(depth, dtype=dtype)
    gamma = np.zeros_like(depth, dtype=dtype)
    alpha[mask] = (w_0[mask] * (x[mask] - center_x) * d[mask] * focal_y) - focal_y
    beta[mask] = w_0[mask] * (y[mask] - center_y) * d[mask] * focal_x
    gamma[mask] = w_0[mask] * d[mask] * (focal_x * focal_y)

    # Compute the coefficients of the second linear equation.
    delta = np.zeros_like(depth, dtype=dtype)
    epsilon = np.zeros_like(depth, dtype=dtype)
    phi = np.zeros_like(depth, dtype=dtype)
    delta[mask] = w_1[mask] * (x[mask] - center_x) * d[mask] * focal_y
    epsilon[mask] = (w_1[mask] * (y[mask] - center_y) * d[mask] * focal_x) - focal_x
    phi[mask] = w_1[mask] * d[mask] * (focal_x * focal_y)

    # Allocate the space for the 3D normals and address them as in the report.
    space_normal = np.zeros((height, width, 3), dtype=dtype)
    a = space_normal[:, :, 0]
    b = space_normal[:, :, 1]
    c = space_normal[:, :, 2]

    # ==== CASE w_0(x, y) NOT ZERO AND w_1(x, y) NOT ZERO ==============================================================

    # Detect the entries associated to the current case.
    mask_case = (w_0 != 0) & (w_1 != 0) & mask

    # Auxiliary variables.
    kappa = np.zeros_like(depth, dtype=dtype)
    alpha_beta_kappa = np.zeros_like(depth, dtype=dtype)
    one_plus_kappa_sq = np.zeros_like(depth, dtype=dtype)
    kappa[mask_case] = (w_1[mask_case] * focal_y) / (w_0[mask_case] * focal_x)
    alpha_beta_kappa[mask_case] = alpha[mask_case] + (beta[mask_case] * kappa[mask_case])
    one_plus_kappa_sq[mask_case] = 1.0 + (kappa[mask_case] ** 2)

    a[mask_case] = - (np.sign(w_0[mask_case]) * np.abs(gamma[mask_case])) / np.sqrt(
        (alpha_beta_kappa[mask_case] ** 2) + ((gamma[mask_case] ** 2) * one_plus_kappa_sq[mask_case]))
    b[mask_case] = kappa[mask_case] * a[mask_case]
    c[mask_case] = - ((alpha[mask_case] * a[mask_case]) + (beta[mask_case] * b[mask_case])) / gamma[mask_case]

    # ==== CASE w_0(x, y) NOT ZERO AND w_1(x, y) EQUAL TO ZERO =========================================================

    # Detect the entries associated to the current case.
    mask_case = (w_0 != 0) & (w_1 == 0) & mask

    a[mask_case] = - (np.sign(w_0[mask_case]) * np.abs(gamma[mask_case])) / np.sqrt(
        (alpha[mask_case] ** 2) + (gamma[mask_case] ** 2))
    c[mask_case] = - (alpha[mask_case] / gamma[mask_case]) * a[mask_case]

    # ==== CASE w_0(x, y) EQUAL TO ZERO AND w_1(x, y) NOT ZERO =========================================================

    # Detect the entries associated to the current case.
    mask_case = (w_0 == 0) & (w_1 != 0) & mask

    b[mask_case] = - (np.sign(w_1[mask_case]) * np.abs(phi[mask_case])) / np.sqrt(
        (epsilon[mask_case] ** 2) + (phi[mask_case] ** 2))
    c[mask_case] = - (epsilon[mask_case] / phi[mask_case]) * b[mask_case]

    # ==== CASE w_0(x, y) EQUAL TO ZERO AND w_1(x, y) EQUAL TO ZERO ====================================================

    # Detect the entries associated to the current case.
    mask_case = (w_0 == 0) & (w_1 == 0) & mask

    c[mask_case] = - 1.0

    # ==================================================================================================================

    # Check the normal orientations ...

    # Compute the cosine of the angle between the 3D normal and the line of sight of the corresponding 3D point.
    rho = np.zeros((height, width), dtype=dtype)
    rho[mask] = d[mask] * (
            ((a[mask] * (x[mask] - center_x)) / focal_x) +
            ((b[mask] * (y[mask] - center_y)) / focal_y) +
            c[mask])

    # Cases:
    # 1. A 3D normal with negative `rho` indicates a 3D point on the side of a plane visible by the camera.
    # 2. A 3D normal with positive `rho` indicates a 3D point on the side of a plane hidden to the camera.
    #    However, it is sufficient to flip the normal orientation in order to associate the point to the side of the
    #    plane visible by the camera.
    # 3. A 3D normal with zero `rho` indicates a 3D point on a plane aligned with the line of sight of the point and
    #    therefore not visible by the camera (regardless of the side of the plane).

    # No normal must be compliant with the case 2.
    assert np.sum(rho > 0) == 0, 'Error in the normal map correction.'

    # Detect the 3D normals whose orientation is not compatible with a visible point (case 3) and set them to zero.
    mask = (rho == 0)
    space_normal[mask] = 0

    return space_normal


def depth2normal(depth: np.array,
                 focal: Tuple[float, float], center: Tuple[float, float],
                 filter_size: int = 7, filter_sigma: float = 5.0) -> np.array:
    """It computes the 3D normals associated to the 3D points described by the input depth map.

    Args:
        depth: depth map, arranged as an `(H, W)` array.
        focal: tuple containing the (pinhole) camera focal length along the `x` and 'y' axes, respectively.
        center: `x` and `y` coordinates of the center of (pinhole) camera center of projection.
        filter_size: height (and width) of the filters.
        filter_sigma: standard deviation (in pixels) of the Gaussian filter underneath the derivative filters.

    Returns:
        The 3D normals associated to the 3D points in the input depth map.
    """

    # Build the vertical (y) derivative filter.
    d_gauss_dy = gauss_filter_deriv_2d(filter_size, filter_sigma)

    # Build the gradient filter.
    grad_filter = d_gauss_dy.T + (1j * d_gauss_dy)
    # The x and y derivative filters are encoded in the real and imaginary parts of the filter.

    # Compute the inverse depth.
    depth_inv = depth2depth_inv(depth, 'inverse_depth')

    # Compute the inverse depth gradient.
    depth_inv_grad = convolve2d(depth_inv, grad_filter, mode='same', boundary='symm')
    depth_inv_grad = np.stack((np.real(depth_inv_grad), np.imag(depth_inv_grad)), axis=2)

    # Convert the inverse depth gradient field to 3D normals.
    normal = plane2space_normal(depth, depth_inv_grad, focal, center)

    return normal


def check_normal(depth: np.array, normal: np.array,
                 focal: Tuple[float, float], center: Tuple[float, float]) -> np.array:
    """It computes the inner product between the 3D point associated to each pixel and the corresponding 3D normal.

    Args:
        depth: depth map, arranged as an `(H, W)` array.
        normal: normal map, arranged as an `(H, W, 3)` array. Normals must be unitary.
        focal: tuple containing the (pinhole) camera focal length along the `x` and 'y' axes, respectively.
        center: `x` and `y` coordinates of the center of (pinhole) camera center of projection.

    Returns:
        The inner product, arranged as an `(H, w)` array, between the 3D point associated to each pixel and the
        corresponding 3D normal. Entries set to zero represent either pixel with no normal available or whose
        corresponding 3D point is not visible by the camera.
    """

    # Define the data type to be used below: 64-bit precision is recommended.
    dtype = np.float64

    # Convert the input depth map to `dtype`.
    d = depth.astype(dtype, copy=False)

    # Depth map dimensions.
    height = depth.shape[0]
    width = depth.shape[1]

    # Build the depth map grid.
    x, y = np.meshgrid(np.arange(width, dtype=dtype), np.arange(height, dtype=dtype))

    # Extract the camera focal lengths and the coordinates of the camera center of projection.
    focal_x, focal_y = focal
    center_x, center_y = center

    # Detect the entries of the grid where the depth is available.
    mask = (depth > 0) & (depth < float('inf'))

    # Name the 3D normal components as in the report.
    a = normal[:, :, 0]
    b = normal[:, :, 1]
    c = normal[:, :, 2]

    # Compute the cosine of the angle between the 3D normal and the line of sight of the corresponding 3D point.
    rho = np.zeros_like(depth, dtype=dtype)
    rho[mask] = d[mask] * (
            ((a[mask] * (x[mask] - center_x)) / focal_x) +
            ((b[mask] * (y[mask] - center_y)) / focal_y) +
            c[mask])

    # Cases:
    # 1. A 3D normal with negative `rho` indicates a 3D point on the side of a plane visible by the camera.
    # 2. A 3D normal with positive `rho` indicates a 3D point on the side of a plane hidden to the camera.
    #    However, it is sufficient to flip the normal orientation in order to associate the point to the side of the
    #    plane visible by the camera.
    # 3. A 3D normal with zero `rho` indicates a 3D point on a plane aligned with the line of sight of the point and
    #    therefore not visible by the camera (regardless of the side of the plane).

    return rho
