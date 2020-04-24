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
import torch
from typing import Union, Tuple


DEFAULT_DEPTH_RANGE = (1e-1, 1e3)


def depth2depth_inv(depth: Union[np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """It computes `1 / depth`.

    It applies the transformation `1 / depth` to the valid entries of `depth`. The remaining entries are set to zero.
    Valid entries of `depth` must belong to the interval `]0, +inf[`.

    Args:
        depth: depth map, arranged as an `(H, W)` array.

    Returns:
        The transformed depth map.
    """

    # Check the class of the input data.
    depth_class = type(depth).__name__

    # Detect the valid entries.
    mask = (depth > 0) & (depth < float('inf'))

    # Select the valid entries.
    selection = depth[mask]

    # Perform the transformation.
    selection = 1.0 / selection

    # Division could lead to non valid entries. Remove them.
    selection[~((selection > 0) & (selection < float('inf')))] = 0

    # Write the transformed depth.
    if depth_class == 'ndarray':

        depth_inv = np.zeros_like(depth)
        depth_inv[mask] = selection

    elif depth_class == 'Tensor':

        # >>> NOT TESTED !!! <<<

        depth_inv = torch.zeros_like(depth)
        depth_inv[mask] = selection

    else:

        raise TypeError('The input must be either of type `numpy.ndarray` or `torch.Tensor`.')

    return depth_inv


def depth_inv2depth(depth_inv: Union[np.array, torch.Tensor],
                    depth_range: Tuple[np.float, np.float] = DEFAULT_DEPTH_RANGE) -> Union[np.array, torch.Tensor]:
    """It reverts the operation of the function `depth2depth_inv()`.

    It reverts the operation of the function `depth2depth_inv()` by applying the transformation `1 / depth_inv`
    to the valid entries of `depth_inv`. The remaining entries are set to zero.
    Valid entries of `depth_inv` must belong to the interval `]0, +inf[`.
    Upon conversion, the valid depth entries are clipped to the interval `[depth_range[0], depth_range[1]]`,
    which must belong to `]0, +inf[`.

    Args:
        depth_inv: transformed depth map, arranged as an `(H, W)` array.
        depth_range: 2-ple specifying the final depth range.

    Returns:
        The depth map resulting from the inverse transformation.
    """

    # Check the class of the input data.
    depth_inv_class = type(depth_inv).__name__

    # Check the final depth range.
    low, up = depth_range
    assert low > 0 and up < float('inf'), 'The depth range must belong to ]0, +inf[.'

    # Detect the valid entries.
    mask = (depth_inv > 0) & (depth_inv < float('inf'))

    # Select the valid entries.
    selection = depth_inv[mask]

    # Perform the transformation.
    selection = 1.0 / selection

    # Division could lead to non valid entries. Remove them.
    selection[~((selection > 0) & (selection < float('inf')))] = 0

    # Clip and write the transformed depth.
    if depth_inv_class == 'ndarray':

        # Clip.
        selection[selection > 0] = np.clip(selection[selection > 0], low, up)

        # Write.
        depth = np.zeros_like(depth_inv)
        depth[mask] = selection

    elif depth_inv_class == 'Tensor':

        # >>> NOT TESTED !!! <<<

        # Clip.
        selection[selection > 0] = torch.clamp(selection[selection > 0], low, up)

        # Write.
        depth = torch.zeros_like(depth_inv)
        depth[mask] = selection

    else:

        raise TypeError('The input must be either of type `numpy.ndarray` or `torch.Tensor`.')

    return depth


def depth_range2depth_inv_range(depth_range: Tuple[float, float]) -> Tuple[float, float]:
    """It converts a depth range into the inverse depth range.

    Args:
        depth_range: 2-tuple specifying the depth range.

    Returns:
        The inverse depth range 2-tuple.
    """

    assert depth_range[0] <= depth_range[1], 'The input depth range is empty.'

    assert depth_range[0] > 0 and depth_range[1] < float('inf'), 'The input depth range must belong to ]0, 1[.'

    return 1.0 / depth_range[1], 1.0 / depth_range[0]


def tensor2array(tensor: torch.Tensor) -> np.array:
    """It converts a torch batch to a numpy batch.

    It converts a batch of images stored as a torch tensor of dimensions `(B, C, H, W)` or `(C, H, W)` into a numpy
    array of dimensions `(B, H, W, C)` or `(H, W, C)`, respectively.

    Args:
        tensor: tensor to convert.

    Returns:
        The converted tensor.
    """

    if tensor.dim() == 3:
        array = np.transpose(tensor.numpy(), (1, 2, 0))
    elif tensor.dim() == 4:
        array = np.transpose(tensor, (0, 2, 3, 1))
    else:
        raise ValueError('Input tensor dimension must be 3 or 4.')

    return array
