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

import argparse
import os
import numpy as np
from cv2 import imread
from iofuns import read_depth_map, read_normal_map, write_bin_file
from misc import depth_percentage_error
from refinement import refine_depth
from logger import Logger
import torch.optim
import time
import math


def read_param() -> argparse.Namespace:
    """It parses the command-line parameters.

    Returns:
        The input parameters.
    """

    # Create the parser.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ========================================== SCALE-INDEPENDENT PARAMETERS ==========================================

    # Input/output paths.
    parser.add_argument(
        '--image', type=str, required=True,
        help='input image path')
    parser.add_argument(
        '--depth', type=str, required=True,
        help='input depth map path')
    parser.add_argument(
        '--normal', type=str,
        help='input normal map path')
    parser.add_argument(
        '--confidence', type=str, required=True,
        help='input (depth) confidence map path')
    parser.add_argument(
        '--depth_gt', type=str,
        help='ground truth depth map path')
    parser.add_argument(
        '--depth_out', type=str, required=True,
        help='refined depth map saving path')
    parser.add_argument(
        '--normal_out', type=str, required=True,
        help='estimated normal map saving path')

    # Camera parameters.
    parser.add_argument(
        '--cam_focal', type=float, nargs=2, required=True,
        help='camera focal length (f_x, f_y)')
    parser.add_argument(
        '--cam_center', type=float, nargs=2, required=True,
        help='camera center of projection (c_x, c_y)')

    # Depth range.
    parser.add_argument(
        '--depth_min', type=float, default=1e-1,
        help='minimum depth value (in meters)')
    parser.add_argument(
        '--depth_max', type=float, default=100,
        help='maximum depth value (in meters)')

    # Confidence binarization.
    parser.add_argument(
        '--confidence_threshold', type=float, default=None,
        help='threshold for confidence binarization')

    # Plotting parameters.
    parser.add_argument(
        '--visdom_display_port', type=int, default=-1,
        help='port to be used by the VISDOM server')
    parser.add_argument(
        '--visdom_base_url', type=str, default='refinement',
        help='string to customize the VISDOM server URL')
    parser.add_argument(
        '--plotting_step', type=int, default=500,
        help='number of steps between two plot updates in the VISDOM server')

    # Device and precision.
    parser.add_argument(
        '--gpu_id', type=int, default=-1,
        help='gpu id (non positive numbers trigger cpu computation)')
    parser.add_argument(
        '--precision', type=str, choices=['single', 'double'], default='single',
        help='computation precision (32 or 64 bits)')

    # Error evaluation.
    parser.add_argument(
        '--depth_error_threshold', type=float, default=0.02,
        help='error threshold (in meters) to be used in the evaluation against the ground truth')

    # Multi-scale strategy.
    parser.add_argument(
        '--scale_nb', type=int, default=1,
        help='number of scales in the multi-scale pyramid')

    # Regularization.
    parser.add_argument(
        '--regularization', type=int, choices=[1], default=1,
        help='regularization type (only regularization of type 1 is available at the moment)')

    # =========================================== SCALE-DEPENDENT PARAMETERS ===========================================

    # Loss parameters.
    parser.add_argument(
        '--lambda_depth_consistency', nargs='*', type=float, default=1.0,
        help='depth consistency term multiplier (one per scale)')
    parser.add_argument(
        '--lambda_normal_consistency', nargs='*', type=float, default=0.0,
        help='normal consistency term multiplier (one per scale)')
    parser.add_argument(
        '--lambda_regularization', nargs='*', type=float, default=7.5,
        help='depth regularization term multiplier (one per scale)')
    parser.add_argument(
        '--gamma_regularization', nargs='*', type=float, default=5.5,
        help='depth regularization term internal multiplier (one per scale)')

    # Graph parameters.
    parser.add_argument(
        '--window_size', nargs='*', type=int, default=9,
        help='search window size (window_size x window_size) to be used in the graph construction (one per scale)')
    parser.add_argument(
        '--patch_size', nargs='*', type=int, default=3,
        help='patch size (patch_size x patch_size) to be used in the graph construction (one per scale)')
    parser.add_argument(
        '--sigma_int', nargs='*', type=float, default=0.07,
        help='color difference standard deviation for patch comparison in the graph construction (one per scale)')
    parser.add_argument(
        '--sigma_spa', nargs='*', type=float, default=3.0,
        help='euclidean distance standard deviation for patch comparison in the graph construction (one per scale)')
    parser.add_argument(
        '--degree_max', nargs='*', type=int, default=20,
        help='maximum number of per pixel neighbors in the graph (one per scale)')

    # Stopping criteria.
    parser.add_argument(
        '--iter_max', nargs='*', type=int, default=4000,
        help='maximum number of iterations (one per scale)')
    parser.add_argument(
        '--eps_stop', nargs='*', type=float, default=1e-6,
        help=('minimum relative change between the current and the previous '
              'iteration depth maps (one per scale)'))
    parser.add_argument(
        '--attempt_max', nargs='*', type=int, default=50,
        help='maximum number of iterations without improving the loss (one per scale)')

    # Learning rate update policies.
    parser.add_argument(
        '--lr_start', nargs='*', type=float, default=1e-4,
        help='initial learning rate (one per scale)')
    parser.add_argument(
        '--lr_slot_nb', nargs='*', type=int, default=1,
        help=('number of partitions (one per scale); '
              'each partition adopts a learning rate which is 1/10 of those employed at the previous partition;'
              '0 excludes the relative depth map change stopping criterium.'))

    # ==================================================================================================================

    # Perform parsing.
    param = parser.parse_args()

    # =================================== CHECK AND ADJUST THE INPUT PARAMETER FORMAT ==================================

    # Cases:
    # - if the value for a parameter is provided, then this must be a list of length `param.scale_nb`.
    # - if the value is not provided, the default value is used to fill a list of length `param.scale_nb`.

    # Check `lambda_depth_consistency`.
    if isinstance(param.lambda_depth_consistency, list):
        assert(len(param.lambda_depth_consistency) == param.scale_nb)
    else:
        param.lambda_depth_consistency = [param.lambda_depth_consistency]*param.scale_nb

    # Check `lambda_normal_consistency`.
    if isinstance(param.lambda_normal_consistency, list):
        assert (len(param.lambda_normal_consistency) == param.scale_nb)
    else:
        param.lambda_normal_consistency = [param.lambda_normal_consistency] * param.scale_nb

    # Check `lambda_reg`.
    if isinstance(param.lambda_regularization, list):
        assert (len(param.lambda_regularization) == param.scale_nb)
    else:
        param.lambda_regularization = [param.lambda_regularization] * param.scale_nb

    # Check `gamma_regularization`.
    if isinstance(param.gamma_regularization, list):
        assert (len(param.gamma_regularization) == param.scale_nb)
    else:
        param.gamma_regularization = [param.gamma_regularization] * param.scale_nb

    # Check `window_size`.
    if isinstance(param.window_size, list):
        assert (len(param.window_size) == param.scale_nb)
    else:
        param.window_size = [param.window_size] * param.scale_nb

    # Check `patch_size`.
    if isinstance(param.patch_size, list):
        assert (len(param.patch_size) == param.scale_nb)
    else:
        param.patch_size = [param.patch_size] * param.scale_nb

    # Check `sigma_int`.
    if isinstance(param.sigma_int, list):
        assert (len(param.sigma_int) == param.scale_nb)
    else:
        param.sigma_int = [param.sigma_int] * param.scale_nb

    # Check `sigma_spa`.
    if isinstance(param.sigma_spa, list):
        assert (len(param.sigma_spa) == param.scale_nb)
    else:
        param.sigma_spa = [param.sigma_spa] * param.scale_nb

    # Check `degree_max`.
    if isinstance(param.degree_max, list):
        assert (len(param.degree_max) == param.scale_nb)
    else:
        param.degree_max = [param.degree_max] * param.scale_nb

    # Check `iter_max`.
    if isinstance(param.iter_max, list):
        assert (len(param.iter_max) == param.scale_nb)
    else:
        param.iter_max = [param.iter_max] * param.scale_nb

    # Check `eps_stop`.
    if isinstance(param.eps_stop, list):
        assert (len(param.eps_stop) == param.scale_nb)
    else:
        param.eps_stop = [param.eps_stop] * param.scale_nb

    # Check `attempt_max`.
    if isinstance(param.attempt_max, list):
        assert (len(param.attempt_max) == param.scale_nb)
    else:
        param.attempt_max = [param.attempt_max] * param.scale_nb

    # Check `lr_start`.
    if isinstance(param.lr_start, list):
        assert (len(param.lr_start) == param.scale_nb)
    else:
        param.lr_start = [param.lr_start] * param.scale_nb

    # Check `lr_slot_nb`.
    if isinstance(param.lr_slot_nb, list):
        assert (len(param.lr_slot_nb) == param.scale_nb)
    else:
        param.lr_slot_nb = [param.lr_slot_nb] * param.scale_nb

    return param


def print_param(param: argparse.Namespace) -> None:
    """It prints the input parameters.

    Args:
        param: parameters to be printed.
    """

    # Organize the parameters into a single string.
    message = ''
    message += '---------------------- Options ----------------------\n'
    for k, v in sorted(vars(param).items()):

        # Turn `v` into a string.
        if isinstance(v, list):
            v_str = ', '.join([str(item) for item in v])
        else:
            v_str = str(v)

        # Write the current pair.
        message += '{:>30}: {:<30}\n'.format(str(k), v_str)
    message += '------------------------ End ------------------------'

    # Print the options to standard output.
    print(message)

    # Save the parameters to disk.
    file_name = 'param.txt'
    with open(file_name, 'wt') as param_file:
        param_file.write(message)
        param_file.write('\n')


def main():

    # Read the input parameters.
    param = read_param()

    # Interrupt the script if the refined depth and normal maps already exist.
    if os.path.exists(param.depth_out) and os.path.exists(param.normal_out):
        print('The refined depth and/or normal map already exist !!!')
        return

    # Organize the camera parameters in a dictionary.
    camera_param = {
        'f_x': param.cam_focal[0],
        'f_y': param.cam_focal[1],
        'c_x': param.cam_center[0],
        'c_y': param.cam_center[1]}

    # Store the loss parameters as a list of dictionaries (one dictionary for each scale of the multi-scale pyramid).
    # The same approach is adopted for the optimization parameters.
    loss_param = [None] * param.scale_nb
    opt_param = [None] * param.scale_nb
    for i in range(param.scale_nb):

        loss_param[i] = {
            'lambda_depth_consistency': param.lambda_depth_consistency[i],
            'lambda_normal_consistency': param.lambda_normal_consistency[i],
            'lambda_regularization': param.lambda_regularization[i],
            'gamma_regularization': param.gamma_regularization[i],
            'window_size': param.window_size[i],
            'patch_size': param.patch_size[i],
            'sigma_intensity': param.sigma_int[i],
            'sigma_spatial': param.sigma_spa[i],
            'degree_max': param.degree_max[i],
            'regularization': param.regularization}

        opt_param[i] = {
            'iter_max': param.iter_max[i],
            'plotting_step': param.plotting_step,
            'eps_stop': param.eps_stop[i],
            'attempt_max': param.attempt_max[i],
            'learning_rate': {'lr_start': param.lr_start[i], 'lr_slot_nb': param.lr_slot_nb[i]},
            'depth_error_threshold': param.depth_error_threshold}

    # Set the device.
    if torch.cuda.is_available() and (param.gpu_id >= 0):
        device = torch.device('cuda:{}'.format(param.gpu_id))
    else:
        device = torch.device('cpu')

    # Create the logger object for plotting.
    logger = None
    if param.visdom_display_port > 0:
        logger = Logger(
            param.depth_error_threshold,
            display_port=param.visdom_display_port, base_url=('/' + param.visdom_base_url))

    ################################################## REFERENCE IMAGE #################################################

    # Read the reference image.
    image = imread(param.image)
    if image is None:
        raise FileNotFoundError('The reference image could not be loaded.')

    # Convert the image to [0, 1] and flip the color channels, as OpenCV assumes that the image is in BGR format on disk.
    image = np.flip(image.astype(param.precision) / 255, axis=2)

    ############################################ NOISY/INCOMPLETE DEPTH MAP ############################################

    # Read the noisy/incomplete depth map.
    depth = read_depth_map(param.depth, 'COLMAP').astype(param.precision)
    if depth is None:
        raise FileNotFoundError('The noisy/incomplete depth map to process could not be loaded.')

    # Clip the valid entries of the MVS depth map to `[param.depth_min, param.depth_max]`.
    mask = (depth > 0) & (depth < float('inf'))
    depth[~mask] = 0    # Non valid pixels are set to zero.
    depth[mask] = np.clip(depth[mask], param.depth_min, param.depth_max)    # Valid pixels are clipped.

    ############################################ NOISY/INCOMPLETE NORMAL MAP ###########################################

    # Read the noisy/incomplete normal map and use them for normal map initialization.
    normal = read_normal_map(param.normal, 'COLMAP')
    if normal is None:
        print('WARNING: The noisy/incomplete normal map could not be loaded.')
    else:
        # Set to zero all the 3D normals without a corresponding depth value.
        normal[~mask] = 0

    ################################################## CONFIDENCE MAP ##################################################

    # Read the confidence map associated to the noisy/incomplete depth map.
    depth_confidence = read_depth_map(param.confidence, 'COLMAP').astype(param.precision)

    # Make the confidence binary.
    if param.confidence_threshold is not None:
        mask_confidence = depth_confidence < param.confidence_threshold
        depth_confidence = np.ones_like(depth_confidence)
        depth_confidence[mask_confidence] = 0

    ############################################## GROUND TRUTH DEPTH MAP ##############################################

    # Read the Ground Truth depth map.
    depth_gt = None
    if param.depth_gt is not None:
        depth_gt = read_depth_map(param.depth_gt, 'COLMAP').astype(param.precision)
        if depth_gt is None:
            raise FileNotFoundError('The ground truth depth map could not be loaded.')

    ####################################################################################################################
    #################################################### REFINEMENT ####################################################
    ####################################################################################################################

    # Start measuring the processing time.
    time_start = time.time()

    # Refine the noisy/incomplete depth map.
    depth_refined, normal_refined = refine_depth(
        image, depth, (param.depth_min, param.depth_max),
        camera_param, loss_param, opt_param,
        depth_confidence=depth_confidence,
        normal=normal,
        depth_gt=depth_gt,
        logger=logger,
        device=device)
    
    # Check the processing time.
    time_elapsed = time.time() - time_start
    minute_elapsed = math.floor(time_elapsed / 60)
    print('Elapsed time: {}min {}sec'.format(
        minute_elapsed, math.ceil(time_elapsed - minute_elapsed * 60)), flush=True)

    #################################################### EVALUATION ####################################################

    # Compute the depth percentage error of the noisy/incomplete depth map.
    if depth_gt is not None:
        print('Percentage of input depth map values with error larger that {}: {:.2f}'.format(
            param.depth_error_threshold,
            depth_percentage_error(depth, depth_gt, param.depth_error_threshold)))

        print('Percentage of refined depth map values with error larger that {}: {:.2f}'.format(
            param.depth_error_threshold,
            depth_percentage_error(depth_refined, depth_gt, param.depth_error_threshold)))

    ###################################################### SAVING ######################################################

    # Save the refined depth map.
    saving_path, _ = os.path.split(param.depth_out)
    os.makedirs(saving_path, exist_ok=True)
    write_bin_file(
        depth_refined, os.path.join(param.depth_out))

    # Save the refined/estimated normal map.
    saving_path, _ = os.path.split(param.normal_out)
    os.makedirs(saving_path, exist_ok=True)
    write_bin_file(
        normal_refined, os.path.join(param.normal_out))


if __name__ == '__main__':
    main()
