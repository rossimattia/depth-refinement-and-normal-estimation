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

import torch
import torch.nn as nn
from misc import similarity_graph, unravel_index
import numpy as np
from typing import Tuple


class DepthConsistencyL1(nn.Module):
    """This class implements a consistency loss between the input depth map and the estimated one. The consistency is
    measured in terms of the L1-norm of the error between the input depth map and the estimated one.
    """

    def __init__(self,
                 depth: np.array, depth_range: Tuple[float, float],
                 depth_confidence: np.array = None,
                 multiplier: float = 0.0):
        """Constructor.

        Args:
            depth: depth map to refine, arranged as an `(H, W)` array.
            depth_range: depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
            depth_confidence: confidence map associated to the depth map to refine. It must have entries in `[0, 1]`.
            multiplier: loss multiplier.
        """

        super(DepthConsistencyL1, self).__init__()

        # Check the input depth range.
        depth_min, depth_max = depth_range
        assert depth_min < depth_max, 'The specified depth range is empty.'

        # Extract the depth map confidence.
        if depth_confidence is not None:
            assert (depth_confidence >= 0).all() and (depth_confidence <= 1).all(), \
                'Depth confidence entries must belong to [0, 1].'
            confidence = depth_confidence
        else:
            confidence = 1

        # The confidence is set to zero at non valid depth entries.
        confidence = confidence * ((depth > depth_min) & (depth < depth_max))

        # Convert the confidence to tensor and register it.
        self.register_buffer('confidence', torch.as_tensor(confidence[None, None, ]))

        # Convert the depth map to tensor and register it.
        self.register_buffer('depth', torch.as_tensor(depth[None, None, ]))

        # Register the normalization constant.
        # self.norm_const = self.confidence.sum()
        pixel_nb = depth.shape[0] * depth.shape[1]
        self.norm_const = pixel_nb

        # Register the loss multiplier.
        self.multiplier = multiplier

    def forward(self, depth: torch.Tensor) -> torch.Tensor:

        # Allocate a zero loss in the case that the loss is disabled, i.e., `self.multiplier` is zero.
        loss = depth.new_zeros(1, requires_grad=True)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:

            # Evaluate the loss.
            loss = (depth - self.depth).mul(self.confidence).abs().sum().div(self.norm_const)

            # Weight the loss.
            loss = self.multiplier * loss

        return loss


class NormalConsistencyL1(nn.Module):
    """This class implements a consistency loss between the input normal map and the estimated one. The consistency is
    measured in terms of the L1-norm of the error between each pair of input and estimated normals.
    """

    def __init__(self,
                 normal: np.array,
                 normal_confidence: np.array = None,
                 multiplier: float = 0.0):
        """Constructor.

        Args:
            normal: 2D or 3D normal map to refine, arranged as an `(H, W, 2)` or `(H, W, 3)` array.
            normal_confidence: confidence map associated to the normal map to refine. It must have entries in `[0, 1]`.
            multiplier: loss multiplier.
        """

        super(NormalConsistencyL1, self).__init__()

        # Extract the normal map confidence.
        if normal_confidence is not None:
            assert (normal_confidence >= 0).all() and (normal_confidence <= 1).all(), \
                'Depth confidence entries must belong to [0, 1].'
            confidence = normal_confidence
        else:
            confidence = 1

        # The confidence is set to zero at non valid normal entries.
        aux = np.sum(normal, axis=2)
        confidence = confidence * ((aux > 0) & (aux < float('inf')))

        # Convert the confidence to tensor and register it.
        self.register_buffer('confidence', torch.as_tensor(confidence[None, None,]))

        # Convert the normal map to tensor and register it.
        self.register_buffer('normal', torch.as_tensor((np.transpose(normal, (2, 0, 1))[None, ]).copy()))

        # Register the normalization constant.
        # self.norm_const = self.confidence.sum()
        pixel_nb = normal.shape[0] * normal.shape[1]
        self.norm_const = pixel_nb

        # Register the loss multiplier.
        self.multiplier = multiplier

    def forward(self, normal: torch.Tensor) -> torch.Tensor:

        # Allocate a zero loss in the case that the loss is disabled, i.e., `self.multiplier` is zero.
        loss = normal.new_zeros(1, requires_grad=True)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:

            # Evaluate the loss.
            loss = (normal - self.normal).mul(self.confidence).abs().sum().div(self.norm_const)

            # Weight the loss.
            loss = self.multiplier * loss

        return loss


class PieceWisePlanarRegularization(nn.Module):
    """This class implements a regularizer promoting piece-wise planar functions.
    """

    def __init__(self,
                 image: np.array,
                 gamma: float,
                 window_size: int = 9, patch_size: int = 7,
                 sigma_intensity: float = 0.2, sigma_spatial: float = 3.0,
                 degree_max: int = 15,
                 version: int = 1,
                 multiplier: float = 0.0,
                 device: torch.device = torch.device('cpu')):
        """Constructor.

        Args:
            image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
            gamma: internal multiplier associated to the vector field part of the loss.
            window_size: search window size (window_size x window_size) to be used in the graph construction.
            patch_size: patch size (patch_size x patch_size) to be used in the graph construction.
            sigma_intensity: color difference standard deviation for patch comparison in the graph construction.
            sigma_spatial: euclidean distance standard deviation for patch comparison in the graph construction.
            degree_max: maximum number of per pixel neighbors in the graph.
            version: regularization version (`0` for NLTGV or `1` for ours).
            multiplier: loss multiplier.
            device: device where the graph computation must take place.
        """

        super(PieceWisePlanarRegularization, self).__init__()

        # Convert the reference image to tensor.
        if image.ndim == 2:
            image_aux = torch.as_tensor(image[None, None, ])
        elif image.ndim == 3:
            image_aux = torch.as_tensor((np.transpose(image, (2, 0, 1))[None, ]).copy())
        else:
            raise ValueError('The input image must be either gray scale or RGB.')

        # Image dimensions.
        height = image_aux.size(2)
        width = image_aux.size(3)

        # Compute the neighboring pixels and the corresponding weights.
        weights, neighbours = similarity_graph(
            image_aux.to(device),
            window_size=window_size, patch_size=patch_size,
            sigma_intensity=sigma_intensity, sigma_spatial=sigma_spatial,
            degree_max=degree_max)
        weights = weights.to('cpu')
        neighbours = neighbours.to('cpu')
        # The function `similarity_graph` is fed with a copy of `image_tensor` on `device`, therefore the output is on
        # `device` as well and it must be brought back to CPU.

        # Register the number of neighbors per pixel.
        self.neighbour_nb = weights.size(1)

        # Flatten the spatial dimensions of `weights` and `neighbours`, and register them.
        weights = weights.view(self.neighbour_nb, -1)
        neighbours = neighbours.view(self.neighbour_nb, -1)
        self.register_buffer('weights', weights)
        self.register_buffer('neighbours', neighbours)

        # Compute the distance vector between each pixel and its neighbours, and register it.
        y_source, x_source = unravel_index(
            torch.arange(height * width).view(1, -1),
            (height, width))
        y_target, x_target = unravel_index(
            self.neighbours,
            (height, width))
        dist = torch.cat(
            (x_source.add(-1, x_target.to(x_source))[:, None, ],
             y_source.add(-1, y_target.to(y_source))[:, None, ]),
            dim=1)
        self.register_buffer('dist', dist.to(torch.float64))
        # Note that `dist` is casted to `torch.float64` before to be registered. In fact, the function `forward()`
        # requires `self.dist` data type to match the float data type (16, 32 or 64) of the other tensors involved
        # in the computation. One could argue that calling `to()` on the module and specifying the data type would
        # convert all its registered tensors. However, this is not the case for integer tensor. Therefore, in order
        # to have `self.dist` converted by `to()`, its data type must be of type float already. The data type
        # `torch.float64` is chosen to avoid any loss of precision.

        # Number of pixels.
        pixel_nb = height * width

        # Register the normalization constant.
        self.norm_const = pixel_nb

        # Register the multiplier associated to the second order derivative.
        self.gamma = gamma

        # Register the regularization type.
        if version == 1:
            self.forward_internal = self.ours
        else:
            raise NotImplementedError('The required regularization does not exist.')

        # Register the loss multiplier.
        self.multiplier = multiplier

    def forward(self, sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:

        return self.forward_internal(sig1, sig2)

    # Our regularization.
    def ours(self, sig1: torch.Tensor, sig2: torch.Tensor) -> torch.Tensor:
        """
        It implements the regularization proposed in the following article:

        Mattia Rossi, Mireille El Gheche, Andreas Kuhn, Pascal Frossard,
        "Joint Graph-based Depth Refinement and Normal Estimation",
        in IEEE Computer Vision and Pattern Recognition Conference (CVPR), Seattle, WA, USA, 2020.

        Args:
            sig1: main signal, arranged as a `(1, 1, H, W)` tensor.
            sig2: secondary signal, arranged as a `(1, 2, H, W)` tensor.

        Returns:
            The considered regularization evaluated at `(sig1, sig2)`.
        """

        # Allocate a zero loss in the case that the loss is disabled, i.e., `self.multiplier` is zero.
        loss = sig1.new_zeros(1, requires_grad=True)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:

            # Expand and flatten `sig1` and `sig2`.
            sig1_flattened = sig1[:, None, ]
            sig1_flattened = sig1_flattened.expand(
                -1, self.neighbour_nb, -1, -1, -1).view(self.neighbour_nb, -1)
            sig2_flattened = sig2[:, None, ]
            sig2_flattened = sig2_flattened.expand(
                -1, self.neighbour_nb, -1, -1, -1).view(self.neighbour_nb, 2, -1)

            # Compute the left part of the regularization.
            aux1 = (sig1_flattened -
                    torch.gather(sig1_flattened, 1, self.neighbours) -
                    (sig2_flattened * self.dist).sum(dim=1))
            aux1 = (aux1 * self.weights).norm(dim=0).sum()

            # Compute the right part of the regularization.
            aux2 = (sig2_flattened -
                    torch.gather(sig2_flattened, 2, self.neighbours[:, None, ].expand(-1, 2, -1))).norm(dim=1)
            aux2 = (aux2 * self.weights).sum()

            # Add the contribution of the left and right parts.
            loss = aux1 + (self.gamma * aux2)

            # Normalize the loss.
            loss = loss.div(self.norm_const)

            # Weight the loss.
            loss = self.multiplier * loss

        return loss
