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
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch import device as dev
import numpy as np
from misc import resize_map, space2plane_normal, plane2space_normal, depth2normal
from filters import gradient_filter
from losses import DepthConsistencyL1, NormalConsistencyL1, PieceWisePlanarRegularization
from cv2 import cvtColor, COLOR_RGB2GRAY
from transforms import depth2depth_inv, depth_inv2depth, depth_range2depth_inv_range
from logger import Logger
from typing import Tuple, List, Dict


class Loss(nn.Module):
    """It creates a loss function consisting of an inverse depth map consistency loss, a 2D normal map consistency loss
    and a joint inverse depth map and normal map regularization. The 2D normal map is 2D vector field capturing the
    orientation of the inverse depth map.

    The independent variables of this loss are `self.idepth` and `self.inormal`.
    """

    def __init__(self,
                 image: np.array, idepth: np.array, idepth_range: Tuple[float],
                 loss_param: Dict[str, float],
                 idepth_confidence: np.array = None,
                 inormal: np.array = None,
                 idepth_init: np.array = None,
                 inormal_init: np.array = None,
                 device: torch.device = torch.device('cpu')) -> None:
        """`Loss` constructor. It considers the inverse depth map and the corresponding 2D normal map.

        Args:
            image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
            idepth: inverse depth map to refine, arranged as an `(H, W)` array.
            idepth_range: inverse depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
            loss_param: dictionaries containing the loss parameters.
            idepth_confidence: confidence map associated to the inverse depth map to refine.
                It must have entries in `[0, 1]`.
            inormal: 2D normal map associated to the depth map to refine, arranged as an `(H, W, 2)` array.
                It is ignored if the normal consistency loss is off.
            idepth_init: initial guess for the refined inverse depth map.
            inormal_init: initial guess for the 2D normal map associated to the refined inverse depth map.
            device: device on which the computation will take place.
        """

        # Call the parent constructor.
        super(Loss, self).__init__()

        # Convert the input data from `np.array` to `torch.Tensor`. In particular, arrays are converted into 4D tensors
        # of size `(1, C, H, W)` with `H`, `W` and `C` representing the height, width and channel number, respectively.

        # Check the inverse depth range and register it.
        if idepth_range[0] <= 0 or idepth_range[1] == float('inf') or idepth_range[0] > idepth_range[1]:
            raise ValueError('Invalid depth range.')
        self.idepth_min = idepth_range[0]
        self.idepth_max = idepth_range[1]

        # Register the first optimization variable, i.e., the refined inverse depth map, and initialize it.
        if idepth_init is not None:
            aux = torch.as_tensor(idepth_init[None, None, ])
        else:
            aux = torch.as_tensor(idepth[None, None, ])
        self.idepth = nn.Parameter(aux.clone(), requires_grad=True)
        # Note that the data passed to `self.idepth` is copied in order to avoid shared data between different tensors.

        # Register the second optimization variable, i.e., the normal map, and initialize it.
        if inormal_init is not None:
            aux = torch.as_tensor((np.transpose(inormal_init, (2, 0, 1))[None, ]).copy())
        elif inormal is not None:
            aux = torch.as_tensor((np.transpose(inormal, (2, 0, 1))[None, ]).copy())
        else:
            with torch.no_grad():
                filter_size = 5
                filter_sigma = 5.0
                grad_filter = gradient_filter(filter_size, filter_sigma)
                pad = tuple([int((filter_size - 1) / 2)] * 4)
                aux = fun.conv2d(
                    fun.pad(self.idepth, pad, mode='replicate'),
                    grad_filter.to(self.idepth))
        self.inormal = nn.Parameter(aux.clone(), requires_grad=True)
        # The `torch.no_grad()` block prevents PyTorch from tracking the operation.

        # Create the depth consistency loss.
        self.idepth_consistency_loss = DepthConsistencyL1(
            idepth, idepth_range,
            depth_confidence=idepth_confidence,
            multiplier=loss_param['lambda_depth_consistency'])

        # Create the 2D normal consistency loss.
        if loss_param['lambda_normal_consistency'] > 0:

            assert inormal is not None, 'Cannot activate the normal consistency term with no input normal map.'

            self.inormal_consistency_loss = NormalConsistencyL1(
                inormal,
                normal_confidence=idepth_confidence,
                multiplier=loss_param['lambda_normal_consistency'])

        else:
            self.inormal_consistency_loss = None

        # Create the depth regularization loss.
        self.regularization_loss = PieceWisePlanarRegularization(
            image,
            loss_param['gamma_regularization'],
            window_size=loss_param['window_size'],
            patch_size=loss_param['patch_size'],
            sigma_intensity=loss_param['sigma_intensity'],
            sigma_spatial=loss_param['sigma_spatial'],
            degree_max=loss_param['degree_max'],
            version=loss_param['regularization'],
            multiplier=loss_param['lambda_regularization'],
            device=device)

    def forward(self) -> Tuple[torch.Tensor, float, float, float]:
        """It evaluates the loss function at (`self.idepth`, `self.inormal`).

        Returns:
            the loss function value, and the value of its two terms, at (`self.idepth`, `self.inormal`).
        """

        # Inverse depth consistency loss.
        idepth_consistency_loss = self.idepth_consistency_loss(self.idepth)

        # 2D normal consistency loss.
        if self.inormal_consistency_loss is not None:
            inormal_consistency_loss = self.inormal_consistency_loss(self.inormal)
        else:
            inormal_consistency_loss = self.idepth.new_zeros(1, requires_grad=True)

        # Regularization loss.
        regularization_loss = self.regularization_loss(self.idepth, self.inormal)

        # Assemble the full loss.
        loss = idepth_consistency_loss + inormal_consistency_loss + regularization_loss

        return loss, idepth_consistency_loss.item(), inormal_consistency_loss.item(), regularization_loss.item()


def refine_depth(image: np.array, depth: np.array, depth_range: Tuple[float, float],
                 camera_param: Dict[str, float], loss_param: List[Dict], opt_param: List[Dict],
                 depth_confidence: np.array = None,
                 normal: np.array = None,
                 depth_init: np.array = None,
                 normal_init: np.array = None,
                 depth_gt: np.array = None,
                 logger: Logger = None,
                 device: dev = dev('cpu')) -> Tuple[np.array, np.array]:
    """It refines the input depth map and estimates the corresponding normal map in a multi-scale fashion.

    It refines the input depth map and estimate the corresponding normal map according to the method described
    in the following article:

    Mattia Rossi, Mireille El Gheche, Andreas Kuhn, Pascal Frossard,
    "Joint Graph-based Depth Refinement and Normal Estimation",
    in IEEE Computer Vision and Pattern Recognition Conference (CVPR), Seattle, WA, USA, 2020.

    If the input depth map comes together with a normal map, the latter can be refined as well (rather than estimated)
    by activating the normal consistency term (not described in the article).

    The `loss_param` input parameter contains a list of dictionaries, one for each scale. Each dictionary must contain
    the following keys:
    - lambda_depth_consistency: depth consistency term multiplier.
    - lambda_normal_consistency: normal consistency term multiplier.
    - lambda_regularization: depth regularization term multiplier.
    - gamma_regularization: depth regularization term internal multiplier.
    - window_size: search window size (window_size x window_size) to be used in the graph construction.
    - patch_size: patch size (patch_size x patch_size) to be used in the graph construction.
    - sigma_intensity: color difference standard deviation for patch comparison in the graph construction.
    - sigma_spatial: euclidean distance standard deviation for patch comparison in the graph construction.
    - degree_max: maximum number of per pixel neighbors in the graph.
    - regularization: regularization type (0 for NLTGV, 4 for our regularization).

    The `opt_param` input parameter contains a list of dictionaries, one for each scale. Each dictionary must contain
    the following keys:
    - iter_max: maximum number of iterations.
    - eps_stop: minimum relative change between the current and the previous iteration depth maps.
    - attempt_max: maximum number of iterations without improving the loss.
    - learning_rate: dictionary containing the following keys:
        - lr_start: initial learning rate.
        - lr_slot_nb: number of partitions; each partition adopts a learning rate which is 1/10 of those employed at
                      the previous partition; 0 excludes the relative depth map change stopping criterium.
    - plotting_step: number of steps between two plot updates of the logger.
    - depth_error_threshold: error threshold (in meters) to be used in the evaluation against the ground truth.

    Args:
        image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
        depth: depth map to refine, arranged as an `(H, W)` array.
        depth_range: depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
        camera_param: dictionary containing `f_x`, `f_y`, `c_x`, `c_y`.
        loss_param: list of dictionaries, each one containing the loss parameters for a given scale.
        opt_param: list of dictionaries, each one containing the solver parameters for a given scale.
        depth_confidence: confidence map associated to the depth map to refine. It must have entries in `[0, 1]`.
        normal: 3D normal map to refine, arranged as an `(H, W, 3)` array. It is ignored if the normal consistency loss is off.
        depth_init: initial guess for the refined depth map.
        normal_init: initial guess for the 3D normal map associated to the refined depth map.
        depth_gt: ground truth depth map, arranged as an `(H, W)` array.
        logger: logger to plot visual results and statistics at runtime.
        device: device on which the computation will take place.

    Returns:
        The refined depth map and the corresponding normal map.
    """

    # Number of scales in the multi-scale pyramid.
    scale_nb = len(opt_param)

    # Allocate the multi-scale pyramid.
    scale_pyramid = [None] * scale_nb
    camera_param_pyramid = [None] * scale_nb
    image_pyramid = [None] * scale_nb
    depth_pyramid = [None] * scale_nb
    depth_confidence_pyramid = [None] * scale_nb
    normal_pyramid = [None] * scale_nb
    depth_init_pyramid = [None] * scale_nb
    normal_init_pyramid = [None] * scale_nb
    depth_gt_pyramid = [None] * scale_nb

    # Build the multi-scale pyramid.
    for i in range(scale_nb):

        if i > 0:

            # Compute the image dimensions for the current scale.
            height = int(round(scale_pyramid[i - 1][0] / 2.0))
            width = int(round(scale_pyramid[i - 1][1] / 2.0))
            scale_pyramid[i] = (height, width)

            # Compute the camera parameters for the current scale.
            x_ratio = scale_pyramid[i][1] / scale_pyramid[i - 1][1]
            y_ratio = scale_pyramid[i][0] / scale_pyramid[i - 1][0]
            camera_param_pyramid[i] = {'f_x': camera_param_pyramid[i - 1]['f_x'] * x_ratio,
                                       'f_y': camera_param_pyramid[i - 1]['f_y'] * y_ratio,
                                       'c_x': camera_param_pyramid[i - 1]['c_x'] * x_ratio,
                                       'c_y': camera_param_pyramid[i - 1]['c_y'] * y_ratio}

            # Downscale the image.
            image_pyramid[i] = resize_map(image_pyramid[i - 1], scale_pyramid[i], order=1)

            # Downscale the noisy/incomplete depth map.
            depth_pyramid[i] = resize_map(depth_pyramid[i - 1], scale_pyramid[i], order=0)

            # Downscale the noisy/incomplete depth map confidence.
            if depth_confidence_pyramid[i - 1] is not None:
                depth_confidence_pyramid[i] = resize_map(depth_confidence_pyramid[i - 1], scale_pyramid[i], order=0)
            else:
                depth_confidence_pyramid[i] = None

            # Downscale the noisy/incomplete normal map.
            if normal_pyramid[i - 1] is not None:
                normal_pyramid[i] = resize_map(normal_pyramid[i - 1], scale_pyramid[i], order=0)

            else:
                normal_pyramid[i] = None

            # Downscale the initial depth map estimate (we need only the lowest scale).
            if depth_init_pyramid[i - 1] is not None:
                depth_init_pyramid[i] = resize_map(depth_init_pyramid[i - 1], scale_pyramid[i], order=0)
                depth_init_pyramid[i - 1] = None
            else:
                depth_init_pyramid[i] = None

            # Downscale the initial normal map estimate (we need only the lowest scale).
            if normal_init_pyramid[i - 1] is not None:
                normal_init_pyramid[i] = resize_map(normal_init_pyramid[i - 1], scale_pyramid[i], order=0)
                normal_init_pyramid[i - 1] = None
            else:
                normal_init_pyramid[i] = None

            # Downscale the ground truth depth map.
            if depth_gt_pyramid[i - 1] is not None:
                depth_gt_pyramid[i] = resize_map(depth_gt_pyramid[i - 1], scale_pyramid[i], order=0)
            else:
                depth_gt_pyramid[i] = None

        else:

            # Store the original image dimensions.
            scale_pyramid[i] = (image.shape[0], image.shape[1])

            # Store the original camera parameters.
            camera_param_pyramid[i] = camera_param

            # The lowest scale hosts the original data.
            image_pyramid[i] = image
            depth_pyramid[i] = depth
            depth_confidence_pyramid[i] = depth_confidence
            normal_pyramid[i] = normal
            depth_init_pyramid[i] = depth_init
            normal_init_pyramid[i] = normal_init
            depth_gt_pyramid[i] = depth_gt

    # Reverse the multi-scale pyramid.
    scale_pyramid.reverse()
    camera_param_pyramid.reverse()
    image_pyramid.reverse()
    depth_pyramid.reverse()
    depth_confidence_pyramid.reverse()
    normal_pyramid.reverse()
    depth_init_pyramid.reverse()        # It contains only the lowest scale.
    normal_init_pyramid.reverse()       # It contains only the lowest scale.
    depth_gt_pyramid.reverse()

    # Perform the multi-scale depth refinement.
    scale_name_pyramid = [None] * scale_nb
    depth_refined_pyramid = [None] * scale_nb
    normal_refined_pyramid = [None] * scale_nb
    for i in range(scale_nb):

        scale_name_pyramid[i] = ('{} ({}x{})'.format(i, scale_pyramid[i][0], scale_pyramid[i][1]))
        print('Processing scale {}'.format(scale_name_pyramid[i]))

        # Setup a new plotting environment.
        if logger is not None:

            if depth_gt_pyramid[i] is not None:
                depth_plotting_range = (np.min(depth_gt_pyramid[i]).item(), np.max(depth_gt_pyramid[i]).item())
            else:
                depth_plotting_range = np.percentile(depth, [5, 95])
            logger.setup(env_name=scale_name_pyramid[i], depth_range=depth_plotting_range)

        # Initialize the next scale with the refined depth map and the corresponding normal map from the previous scale.
        # The two maps are up-sampled first.
        if i > 0:
            depth_init_pyramid[i] = resize_map(depth_refined_pyramid[i - 1], scale_pyramid[i], order=0)
            if normal_refined_pyramid[i - 1] is not None:
                normal_init_pyramid[i] = resize_map(normal_refined_pyramid[i - 1], scale_pyramid[i], order=0)

        # Refine the depth map of the current scale.
        depth_refined, normal_refined = refine(
            image_pyramid[i], depth_pyramid[i], depth_range,
            camera_param_pyramid[i], loss_param[i], opt_param[i],
            depth_confidence=depth_confidence_pyramid[i],
            depth_init=depth_init_pyramid[i],
            normal=normal_pyramid[i],
            normal_init=normal_init_pyramid[i],
            depth_gt=depth_gt_pyramid[i],
            logger=logger,
            device=device)

        depth_refined_pyramid[i] = depth_refined
        normal_refined_pyramid[i] = normal_refined

    # Extract the refined depth map and the corresponding normal map.
    depth_refined = depth_refined_pyramid[-1]
    normal_refined = normal_refined_pyramid[-1]

    # Delete all the plotting environments.
    if logger is not None:
        for i in range(scale_nb):
            logger.vis.delete_env(scale_name_pyramid[i])

    return depth_refined, normal_refined


def refine(image: np.array, depth: np.array, depth_range: Tuple[float, float],
           camera_param: Dict[str, float], loss_param: Dict, opt_param: Dict,
           depth_confidence: np.array = None,
           normal: np.array = None,
           depth_init: np.array = None,
           normal_init: np.array = None,
           depth_gt: np.array = None,
           logger: Logger = None,
           device: dev = dev('cpu')) -> Tuple[np.array, np.array]:
    """It implements one scale of the multi-scale pyramid of the function `refine_depth`.

    Args:
        image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
        depth: depth map to refine, arranged as an `(H, W)` array.
        depth_range: depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
        camera_param: dictionary containing `f_x`, `f_y`, `c_x`, `c_y`.
        loss_param: dictionary containing the loss parameters.
        opt_param: dictionary containing the solver parameters.
        depth_confidence: confidence map associated to the depth map to refine. It must have entries in `[0, 1]`.
        normal: 3D normal map to refine, arranged as an `(H, W, 3)` array. It is ignored if the normal consistency loss is off.
        depth_init: initial guess for the refined depth map.
        normal_init: initial guess for the 3D normal map associated to the refined depth map.
        depth_gt: ground truth depth map, arranged as an `(H, W)` array.
        logger: logger to plot visual results and statistics at runtime.
        device: device on which the computation will take place.

    Returns:
        The refined depth map and the corresponding normal map.
    """

    # Check the depth map data type.
    if depth.dtype == np.float32:
        depth_dtype = torch.float
    elif depth.dtype == np.float64:
        depth_dtype = torch.double
    else:
        raise TypeError('The input depth map must be either of type double or float.')

    # Convert the reference image to gray scale.
    image_gray = image
    if image_gray.ndim == 3:
        image_gray = cvtColor(image_gray.astype(np.float32), COLOR_RGB2GRAY)
        image_gray = image_gray.astype(image.dtype)
        # The function `cvtColor` requires an input image of type uint8, uint16 or float32. Therefore, `image_gray` is
        # first converted to float32 (to minimize the precision loss) and then back to its original data type.

    # Plot.
    if logger is not None:
        logger.plot(
            texture=image,
            depth=depth,
            depth_init=depth_init,
            depth_gt=depth_gt,
            normal=normal,
            normal_init=normal_init)

    # Convert the depth maps.
    idepth = depth2depth_inv(depth)
    idepth_init = depth2depth_inv(depth_init) if depth_init is not None else None
    idepth_range = depth_range2depth_inv_range(depth_range)

    # Convert the normal maps.
    inormal = None
    inormal_init = None
    if normal is not None:
        inormal = space2plane_normal(
            depth,
            normal,
            (camera_param['f_x'], camera_param['f_y']),
            (camera_param['c_x'], camera_param['c_y']))
    if normal_init is not None:
        inormal_init = space2plane_normal(
            depth_init if depth_init is not None else depth,
            normal_init,
            (camera_param['f_x'], camera_param['f_y']),
            (camera_param['c_x'], camera_param['c_y']))

    # Create the loss object.
    loss = Loss(image_gray, idepth, idepth_range,
                loss_param,
                idepth_confidence=depth_confidence,
                inormal=inormal,
                idepth_init=idepth_init,
                inormal_init=inormal_init,
                device=device).to(device=device, dtype=depth_dtype)

    # Set the maximum number of iterations.
    assert 'iter_max' in opt_param, 'Missing \'iter_max\' in `opt_param`.'
    iter_max = opt_param['iter_max']

    # Set the learning rate and define the optimization policy (i.e., with oir without scheduler).
    assert 'learning_rate' in opt_param, 'Missing \'learning_rate\' in `opt_param.'
    assert 'lr_start' in opt_param['learning_rate'], 'Missing \'lr\' in `opt_param[\'learning_rate\']`.'
    assert 'lr_slot_nb' in opt_param['learning_rate'], 'Missing \'slot_nb\' in `opt_param[\'learning_rate\']`.'
    learning_rate_start = opt_param['learning_rate']['lr_start']
    learning_rate_slot_nb = opt_param['learning_rate']['lr_slot_nb']

    # Define stopping condition.
    if learning_rate_slot_nb < 1:

        # The learning rate is kept constant.

        # The optimization terminates in one of the following event occurs:
        # - the relative depth change is smaller than `eps_stop`,
        # - the loss is not improved for more than `attempt_max` consecutive iterations,
        # - `iter_max` iterations have been performed.

        assert 'eps_stop' in opt_param, 'Missing \'eps_stop\' in `opt_param.'
        assert 'attempt_max' in opt_param, 'Missing \'attempt_max\' in `opt_param.'

        eps_stop = opt_param['eps_stop']
        attempt_max = opt_param['attempt_max']
        scheduler_step_size = iter_max * 2

    else:

        # The learning rate is dynamically updated.

        # The optimization terminates only when `iter_max` iterations have been performed.
        # However, in this scenario the learning rate is progressively decreased:
        # - the learning rate starts at `learning_rate_start`,
        # - it is decreased `learning_rate_slot_nb - 1` times by a factor `10`.

        eps_stop = 0.0
        attempt_max = float('inf')
        scheduler_step_size = int(math.ceil(float(iter_max) / float(learning_rate_slot_nb)))

    # Set the plotting step.
    assert 'plotting_step' in opt_param, 'Missing \'plotting_step\' in `opt_param.'
    plotting_step = opt_param['plotting_step']

    # Allocate an array to store the loss function values.
    loss_history = np.zeros(iter_max + 1)
    idepth_consistency_history = np.zeros(iter_max + 1)
    inormal_consistency_history = np.zeros(iter_max + 1) if loss_param['lambda_normal_consistency'] > 0 else None
    regularization_history = np.zeros(iter_max + 1)

    # Create an ADAM optimizer.
    optimizer = torch.optim.Adam(loss.parameters(), lr=learning_rate_start)

    # Create a learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, gamma=0.1)

    ####################################################################################################################
    ################################################# OPTIMIZATION #####################################################
    ####################################################################################################################

    # Lowest minimum value of the loss encountered during the optimization.
    loss_value_min = float('inf')

    # Number of consecutive iterations without improving `loss_value_min`.
    attempt_counter = 0

    # Relative change of the depth map between two consecutive iterations.
    relative_depth_change = float('inf')

    ################################################# CASE `i == 0` ####################################################

    # Evaluate the loss function.
    optimizer.zero_grad()
    loss_value, idepth_consistency_value, inormal_consistency_value, regularization_value = loss.forward()

    # Log operations.
    with torch.no_grad():

        # Store the current value of the loss.
        idepth_consistency_history[0] = idepth_consistency_value
        if inormal_consistency_history is not None:
            inormal_consistency_history[0] = inormal_consistency_value
        regularization_history[0] = regularization_value
        loss_history[0] = loss_value.item()

        # Log the optimization status to the standard output.
        print('Iteration: {:6}, Fails: {:3}, Rel. depth change: {:.6f}, Loss: {:.6f}'.format(
            0, attempt_counter, relative_depth_change, loss_history[0]), flush=True)

        # Plot the optimization status.
        indexes = np.arange(0, 1)
        if logger is not None:
            depth_aux = depth_inv2depth(
                loss.idepth.data.to('cpu').squeeze().numpy(), depth_range)
            normal_aux = plane2space_normal(
                depth_aux,
                np.transpose(loss.inormal.data.to('cpu').squeeze().numpy(), (1, 2, 0)),
                (camera_param['f_x'], camera_param['f_y']),
                (camera_param['c_x'], camera_param['c_y']))
            logger.plot(
                depth_refined=depth_aux,
                normal_refined=normal_aux,
                idepth_consistency_loss=(indexes, idepth_consistency_history[indexes]),
                inormal_consistency_loss=((indexes, inormal_consistency_history[indexes])
                                          if inormal_consistency_history is not None else None),
                regularization_loss=(indexes, regularization_history[indexes]),
                global_loss=(indexes, loss_history[indexes]))

    ################################################# CASE `i > 0` #####################################################

    for i in range(1, iter_max + 1):

        # Compute the gradient of each parameter of the loss (i.e., the depth map and the normal maps).
        loss_value.backward()

        # Store a copy of the old depth map.
        idepth_old = loss.idepth.clone().detach()

        # Update the old depth map.
        optimizer.step()

        # Update the optimizer learning rate.
        scheduler.step()

        # Without PyTorch tracking, project the new depth map into the specified depth range.
        with torch.no_grad():
            loss.idepth.data = loss.idepth.data.clamp(idepth_range[0], idepth_range[1])

        # Evaluate the loss function at the new depth map and normal map.
        optimizer.zero_grad()
        loss_value, idepth_consistency_value, inormal_consistency_value, regularization_value = loss.forward()

        # Without PyTorch tracking, perform some routines.
        with torch.no_grad():

            # Store the value of the loss evaluated at the new depth map.
            idepth_consistency_history[i] = idepth_consistency_value
            if inormal_consistency_history is not None:
                inormal_consistency_history[i] = inormal_consistency_value
            regularization_history[i] = regularization_value
            loss_history[i] = loss_value.item()

            # Compute the relative depth map change.
            relative_depth_change = torch.norm(
                (idepth_old - loss.idepth).view(-1, 1)) / torch.norm(idepth_old.view(-1, 1))

            # Update the lowest encountered minimum.
            if loss_history[i] >= loss_value_min:
                attempt_counter = attempt_counter + 1
            else:
                attempt_counter = 0
                loss_value_min = loss_history[i]

            # Evaluate the stopping condition.
            stop_now = (relative_depth_change <= eps_stop) or (attempt_counter >= attempt_max)

            if (i % plotting_step == 0) or stop_now or ((i + 1) > iter_max):

                # Log the optimization status to the standard output.
                print('Iteration: {:6}, Fails: {:3}, Rel. depth change: {:.6f}, Loss: {:.6f}'.format(
                    i, attempt_counter, relative_depth_change, loss_history[i]), flush=True)

                # Plot the optimization status.
                indexes = np.arange(i - (plotting_step - 1), i + 1)     # The index `i` is included.
                if logger is not None:
                    depth_aux = depth_inv2depth(
                        loss.idepth.data.to('cpu').squeeze().numpy(), depth_range)
                    normal_aux = plane2space_normal(
                        depth_aux,
                        np.transpose(loss.inormal.data.to('cpu').squeeze().numpy(), (1, 2, 0)),
                        (camera_param['f_x'], camera_param['f_y']),
                        (camera_param['c_x'], camera_param['c_y']))
                    logger.plot(
                        depth_refined=depth_aux,
                        normal_refined=normal_aux,
                        idepth_consistency_loss=(indexes, idepth_consistency_history[indexes]),
                        inormal_consistency_loss=((indexes, inormal_consistency_history[indexes])
                                                  if inormal_consistency_history is not None else None),
                        regularization_loss=(indexes, regularization_history[indexes]),
                        global_loss=(indexes, loss_history[indexes]))

                # If the stopping condition is met, terminate.
                if stop_now:
                    break

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # Extract the refined depth map.
    depth_refined = depth_inv2depth(
        loss.idepth.detach().to('cpu').numpy().squeeze(), depth_range)

    # Extract the normal map associated to the refined depth map.
    normal_refined = plane2space_normal(
        depth_refined,
        np.transpose(loss.inormal.detach().to('cpu').numpy().squeeze(), (1, 2, 0)),
        (camera_param['f_x'], camera_param['f_y']),
        (camera_param['c_x'], camera_param['c_y']))

    return depth_refined, normal_refined
