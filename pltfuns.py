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
# import matplotlib.cm
from typing import Tuple


def normal2rgb(normal: np.array) -> np.array:
    """It maps a 3D normal map into an RGB image.

    It maps the input 3D normal map into an RGB image. Since a normal vector has unitary norm, the set of all the
    possible normals describes a unitary sphere. This function maps each point `(X, Y, Z)` on the sphere, hence each
    normal vector, to an RGB value. All non zero normals are assumed valid and no check is performed on them.

    Args:
        normal: normal map, arranged as an `(H, W, 3)` array.

    Returns:
        An RGB image, arranged as an `(H, W, 3)` array, that encodes the normals.
    """

    # Detect the entries of the grid where the 3D normals are available.
    mask = (np.sum(normal != 0, axis=2) != 0)

    # Allocate the RGB representation of the normals.
    normal_rgb = np.zeros_like(normal, dtype=np.uint8)

    # Map the X, Y and Z coordinates from [-1, 1] to [0, 255].
    normal_rgb[mask] = np.round(((normal.astype(np.float64, copy=False)[mask] + 1.0) / 2.0) * 255).astype(np.uint8)

    return normal_rgb


def normal2rgb_legend(n: int = 500) -> Tuple[np.array, np.array]:
    """It returns a legend for the function `normals2rgb`.

    It returns a legend for the color coding adopted in the function `normal2rgb`. The legend comprises two images
    representing the two hemispheres associated to the negative and positive `Z` semi-axis, respectively.

        Args:
            n: height (and width) of the output legend images.

        Returns:
            The legend arranged as two `(n, n)` arrays, for the negative and positive `Z` semi-axes, respectively.
        """

    # Build the X and Y components of the 3D normals.
    x, y = np.meshgrid(np.linspace(- 1, 1, n), np.linspace(- 1, 1, n))

    # Detect the entries that are within the unitary circle.
    mask = np.sqrt((x ** 2) + (y ** 2)) <= 1.0

    # Compute the z component of the 3D unitary normals.
    z = np.zeros_like(x)
    z[mask] = np.sqrt(np.abs(1 - (x[mask] ** 2) - (y[mask] ** 2)))

    # Set the X and Y entries of the non unitary 3D normals to zero.
    x[~mask] = 0
    y[~mask] = 0

    # Build the negative hemisphere of the 3D normal legend.
    normal_z_neg = np.stack((x, y, - z), axis=2)

    # Build the positive hemisphere of the 3D normal legend.
    normal_z_pos = np.stack((x, y, z), axis=2)

    # Encode the 3D normals into an RGB image.
    normal_z_neg_rgb = normal2rgb(normal_z_neg)
    normal_z_pos_rgb = normal2rgb(normal_z_pos)

    return normal_z_neg_rgb, normal_z_pos_rgb


# def plot_map(heat_map, mask=None, vmax=0.0, vmin=1.0, colormap='viridis'):
#     """It turns the input heat map into an RGB image.
#
#     It turns the input heat map into an RGB image according to the specified input color map. The parameters `vmin`
#     and `vmax` play the same role that they have in `matplotlib.pyplot.imshow`. In particular, calling `imshow` on
#     the input heat map using `vmin` and `vmax` produces the same visual result of calling `imshow` on the RGB image
#     created by this function.
#
#     In addition, the heat map pixels marked as `False` in the input `mask` are converted to white in the RGB image.
#
#     Args:
#         heat_map: heat map, arranged as an `(H, W)` array.
#         mask: binary mask, arranged as an `(H, W)` array.
#         vmax: heat map lower bound.
#         vmin: heat map upper bound.
#         colormap: `matplotlib` colormap.
#
#     Returns:
#         The input heat map converted to RGB.
#     """
#
#     # Clip the input heat map.
#     heat_map_clipped = np.clip(heat_map, vmin, vmax)
#
#     # Color map object.
#     cmap = matplotlib.cm.get_cmap(colormap)
#
#     # Convert the heat map intensity values to RGB triplets.
#     heat_map_rgb = cmap((heat_map_clipped - vmin) / (vmax - vmin))[:, :, 0:-1]
#
#     # Non valid pixels are assigned the white color.
#     if mask is not None:
#         mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
#         heat_map_rgb[~mask_rgb] = 1.0
#
#     return heat_map_rgb
