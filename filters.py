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
from typing import Tuple, Union


def gauss_filter_1d(length: int, sigma: float) -> np.array:
    """It builds a 1D Gaussian filter.

    Args:
        length: number of filter taps.
        sigma: standard deviation.

    Returns:
        A 1D Gaussian filter arranged as a `(length,)` array.
    """

    # Check the filter length.
    if (length % 2) == 0:
        raise ValueError('The length of the filter must be odd.')

    # Build the filter.
    radius = int((length - 1) / 2.0)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    y = np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))

    # Normalize the filter.
    const = np.sum(y)
    assert const != 0, 'The filter is zero everywhere.'
    y = y / const

    return y


def gauss_filter_deriv_1d(length: int, sigma: float) -> np.array:
    """It builds the derivative of a 1D Gaussian filter.

    Args:
        length: number of filter taps.
        sigma: standard deviation.

    Returns:
        A 1D Gaussian filter derivative, arranged as a `(length,)` array.
    """

    # Check the filter length.
    if (length % 2) == 0:
        raise ValueError('The length of the filter must be odd.')

    # Build the filter.
    radius = int((length - 1) / 2.0)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    y = gauss_filter_1d(length, sigma) * (- x / (sigma ** 2))

    # Normalize the filter.
    const = np.sum(np.abs(y))
    assert const != 0, 'The filter is zero everywhere.'
    y = y / const
    # TODO: check whether this normalization makes sense.

    return y


def gauss_filter_2d(size: int, sigma: float) -> np.array:
    """It builds a 2D Gaussian filter.

    Args:
        size: height (and width) of the filter.
        sigma: standard deviation (in pixels) of the Gaussian filter.

    Returns:
        A 2D Gaussian filter arranged as a `(size, size)` array.
    """

    # Build the filter.
    y = (gauss_filter_1d(size, sigma)[:, None]).dot(gauss_filter_1d(size, sigma)[None, :])

    # Normalize the filter.
    const = np.sum(y)
    assert const != 0, 'The filter is zero everywhere.'
    y = y / const

    return y


def gauss_filter_deriv_2d(size: int, sigma: float) -> np.array:
    """It builds the vertical derivative of a 2D Gaussian filter.

    It builds the vertical derivative of a 2D Gaussian filter. The horizontal derivative can be obtained just by taking
    the transpose of the vertical one.

    Args:
        size: height (and width) of the filters.
        sigma: standard deviation (in pixels) of the Gaussian filter underneath the derivative filters.

    Returns:
        The vertical derivative of a 2D Gaussian filter arranged as a `(size, size)` array.
    """

    # Build the filter.
    y = (gauss_filter_deriv_1d(size, sigma)[:, None]).dot(gauss_filter_1d(size, sigma)[None, :])

    # Normalize the filter.
    const = np.sum(np.abs(y))
    assert const != 0, 'The filter is zero everywhere.'
    y = y / const

    return y


def gradient_filter(size: int, sigma: float) -> torch.Tensor:
    """It builds a gradient filter for images in PyTorch tensor format.

    It builds a filter that can be used with `torch.nn.functional.conv2d` to compute the gradient of a batch of images
    or, more in general, of maps. The images or maps must have only one channel.
    The filter is arranged as a `(2, 1, H, W)` tensor with `[0, :, :, :]` and `[1, :, :, :]` the 2D horizontal and
    vertical derivative filters.

    Example:
        batch_nb = 5
        height = 100
        width = 200
        size = 7
        image = torch.random(batch_nb, 1, height, width)
        filter = gradient_filter(7, 0.1)
        pad = tuple([int((size - 1) / 2)] * 4)
        image_grad = torch.nn.functional.conv2d(torch.nn.functional.pad(image, pad, mode='replicate'), filter)

    In the example, `image_grad` is a `(batch_nb, 2, height, width)` tensor with `image_grad[k, 0, :, :]` and
    `image_grad[k, 1, :, :]` the horizontal and vertical derivatives of the image `k`.

    Args:
        size: height (and width) of the filters.
        sigma: standard deviation (in pixels) of the Gaussian filter underneath the derivative filters.

    Returns:
        The gradient filter, arranged as a `(2, 1, H, W)` tensor.
    """

    # Build the vertical (y) derivative filter.
    d_gauss_dy = gauss_filter_deriv_2d(size, sigma)

    # Flip the filter around the (x, y) origin, as torch.nn.functional.conv2d() performs just cross-correlation rather
    # than the standard convolution.
    d_gauss_dy = np.fliplr(d_gauss_dy)
    d_gauss_dy = np.flipud(d_gauss_dy)

    # Build the horizontal (x) derivative filter, which is just the transpose of the vertical one.
    d_gauss_dx = d_gauss_dy.T

    # Expand the filters to make them compliant with torch.nn.functional.conv2d.
    d_gauss_dy = d_gauss_dy[None, None, :, :]  # [1, 1, size, size]
    d_gauss_dx = d_gauss_dx[None, None, :, :]  # [1, 1, size, size]

    # Concatenate the two filters into a single filter with two channels.
    grad_filter = np.concatenate((d_gauss_dx, d_gauss_dy), axis=0)  # [2, 1, size, size]

    # Change the filter type to torch.Tensor.
    grad_filter = torch.from_numpy(grad_filter)

    return grad_filter


def diff_filter_bank(size: Union[int, Tuple[int, int]] = 5):
    """It builds a derivative filter bank.

    It builds a set of `HxW` filters where each filter has only two non zero entries: the central one, whose value
    is `-1`, and another non central, whose value is `1`. The number of filters is `H*W - 1`, i.e., all the possible
    filters of the described type.

    Args:
        size: tuple specifying the height and width of the filter (square filter if only one dimensions is specified).

    Returns:
        The derivative filter bank, arranged as an `(H, W)` array.
    """

    # Filter bank spatial dimensions.
    filter_size = tuple((size, ))
    if len(filter_size) == 2:
        filter_height = size[0]
        filter_width = size[1]
    elif len(filter_size) == 1:
        filter_height = size
        filter_width = size
    else:
        raise TypeError('Input must be either an integer or a 2-tuple of integers.')

    # Number of filters in the filter bank.
    filter_nb = int((filter_height * filter_width) - 1)

    # Center of each filter in the filter bank.
    filter_center_y = int((filter_height - 1) / 2.0)
    filter_center_x = int((filter_width - 1) / 2.0)

    # Create the filter bank.
    index = 0
    filter_bank = torch.zeros(filter_nb, 1, size, size)
    filter_bank[:, :, filter_center_y, filter_center_x] = - 1.0
    for y in range(size):
        for x in range(size):

            if y != filter_center_y or x != filter_center_x:
                filter_bank[index, :, y, x] = 1.0
                index += 1

    return filter_bank
