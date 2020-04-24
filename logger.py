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
import numpy as np
from importlib import import_module
from misc import resize_map, depth_percentage_error
from pltfuns import normal2rgb
from typing import Tuple


# Maximum height of any image (not heat map) plotted on the screen.
HEIGHT_MAX = int(300)


class Logger:

    def __init__(self,
                 error_threshold: float,
                 display_port: int = 8097, base_url: str = '/1234'):

        # Windows.
        self.texture_win = None                 # reference image.
        self.depth_win = None                   # noisy and possibly incomplete depth map.
        self.depth_init_win = None              # initial depth map used in the refinement.
        self.depth_refined_win = None           # refined depth map.
        self.normal_win = None                  # noisy and possibly incomplete normal map.
        self.normal_init_win = None             # initial normal map used in the refinement.
        self.normal_refined_win = None          # normal map associated to the refined depth map.
        self.depth_gt_win = None                # ground truth depth map.
        self.depth_error_win = None             # noisy and possibly incomplete depth map error.
        self.depth_refined_error_win = None     # refined depth map error.

        # Windows associated to the partial and global losses (in the inverse depth domain).
        self.idepth_consistency_loss_win = None
        self.inormal_consistency_loss_win = None
        self.regularization_loss_win = None
        self.global_loss_win = None

        # Typically, the logger is called to plot a new complete depth map and to compute its error with respect to
        # the ground truth depth map, which does not change very often. Therefore, we store the ground truth depth map.
        self.depth_gt = None

        # Minimum and maximum depth values to be used in the plots.
        self.depth_min = None
        self.depth_max = None

        # Record the error threshold to be used in the percentage error computation.
        self.depth_error_threshold = error_threshold

        # Instantiate the online visualization tool.
        visdom = import_module('visdom')
        self.vis = visdom.Visdom(port=display_port, base_url=base_url)
        # The `visdom` module is imported here to avoid its installation in the case the user does not need the logger.

        # Environment default name.
        self.environment = 'main'

    def plot(self,
             texture: np.array = None,
             depth: np.array = None,
             depth_init: np.array = None,
             depth_refined: np.array = None,
             depth_gt: np.array = None,
             normal: np.array = None,
             normal_init: np.array = None,
             normal_refined: np.array = None,
             idepth_consistency_loss: np.array = None,
             inormal_consistency_loss: np.array = None,
             regularization_loss: np.array = None,
             global_loss: np.array = None) -> None:

        # ==============================================================================================================

        # Reference camera texture.
        if texture is not None:

            # Texture dimensions.
            aux = texture
            if texture.ndim == 2:
                height = texture.shape[0]
                width = texture.shape[1]
                aux = aux[:, :, None]
            elif texture.ndim == 3 and texture.shape[2] == 3:
                height = texture.shape[0]
                width = texture.shape[1]
            else:
                raise ValueError('The input texture must be gray scale or RGB.')

            # Resize the texture if too large.
            img_ratio = float(width) / float(height)
            if height > HEIGHT_MAX:
                aux = resize_map(aux, [HEIGHT_MAX, int(HEIGHT_MAX * img_ratio)])

            # Convert the texture to tensor.
            texture_t = torch.from_numpy(np.transpose(aux, axes=(2, 0, 1)).copy())

            # Plot the texture.
            if self.texture_win is not None:

                self.vis.image(
                    texture_t,
                    env=self.environment,
                    win=self.texture_win,
                    opts=dict(title='texture'))

            else:

                self.texture_win = self.vis.image(
                    texture_t,
                    env=self.environment,
                    opts=dict(title='texture'))

        # ==============================================================================================================

        # Ground truth depth map.
        if depth_gt is not None:

            # Store the ground truth depth map.
            self.depth_gt = depth_gt

            # Set the minimum and maximum depth values.
            xmin = self.depth_min if self.depth_min is not None else np.min(self.depth_gt)
            xmax = self.depth_max if self.depth_max is not None else np.max(self.depth_gt)

            # Convert the ground truth depth map to tensor.
            depth_gt_t = torch.from_numpy(self.depth_gt).flip([0])

            # Plot.
            if self.depth_gt_win is not None:

                self.vis.heatmap(
                    depth_gt_t,
                    env=self.environment,
                    win=self.depth_gt_win,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='depth gt'))

            else:

                self.depth_gt_win = self.vis.heatmap(
                    depth_gt_t,
                    env=self.environment,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='depth gt'))

        # ==============================================================================================================

        # Noisy and possibly incomplete depth map.
        if depth is not None:

            # Set the minimum and maximum depth values.
            aux_min, aux_max = np.percentile(depth, [5, 95])
            xmin = self.depth_min if self.depth_min is not None else aux_min
            xmax = self.depth_max if self.depth_max is not None else aux_max

            # Convert the depth map to tensor.
            depth_t = torch.from_numpy(depth).flip([0])

            # Plot.
            if self.depth_win is not None:

                self.vis.heatmap(
                    depth_t,
                    env=self.environment,
                    win=self.depth_win,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='input depth'))

            else:

                self.depth_win = self.vis.heatmap(
                    depth_t,
                    env=self.environment,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='input depth'))

            # Error map.
            if self.depth_gt is not None:

                # Detect the valid entries in `self.depth_gt`.
                mask = (self.depth_gt > 0) & (self.depth_gt < float('inf'))

                # Compute the error.
                error = np.abs(self.depth_gt - depth)
                error[~mask] = 0

                # Compute the percentage error.
                percentage_error = depth_percentage_error(
                    depth, self.depth_gt, self.depth_error_threshold)

                # Convert the error to tensor.
                error_t = torch.from_numpy(error).flip([0])

                # Plot the depth map error.
                if self.depth_error_win is not None:

                    self.vis.heatmap(
                        error_t,
                        env=self.environment,
                        win=self.depth_error_win,
                        opts=dict(
                            xmin=0.0,
                            xmax=self.depth_error_threshold,
                            title='input depth error: {:.2f}% ({})'.format(
                                percentage_error, self.depth_error_threshold)))

                else:

                    self.depth_error_win = self.vis.heatmap(
                        error_t,
                        env=self.environment,
                        opts=dict(
                            xmin=0.0,
                            xmax=self.depth_error_threshold,
                            title='input depth error: {:.2f}% ({})'.format(
                                percentage_error, self.depth_error_threshold)))

        # ==============================================================================================================

        # Initial depth map.
        if depth_init is not None:

            # Set the minimum and maximum depth values.
            aux_min, aux_max = np.percentile(depth_init, [5, 95])
            xmin = self.depth_min if self.depth_min is not None else aux_min
            xmax = self.depth_max if self.depth_max is not None else aux_max

            # Convert the depth map to tensor.
            depth_init_t = torch.from_numpy(depth_init).flip([0])

            # Plot the depth map.
            if self.depth_init_win is not None:

                self.vis.heatmap(
                    depth_init_t,
                    env=self.environment,
                    win=self.depth_init_win,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='initial depth'))

            else:

                self.depth_init_win = self.vis.heatmap(
                    depth_init_t,
                    env=self.environment,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='initial depth'))

        # ==============================================================================================================

        # Refined depth map.
        if depth_refined is not None:

            # Set the minimum and maximum depth values.
            aux_min, aux_max = np.percentile(depth_refined, [5, 95])
            xmin = self.depth_min if self.depth_min is not None else aux_min
            xmax = self.depth_max if self.depth_max is not None else aux_max

            # Convert the depth map to tensor.
            depth_refined_t = torch.from_numpy(depth_refined).flip([0])

            # Plot the depth map.
            if self.depth_refined_win is not None:

                self.vis.heatmap(
                    depth_refined_t,
                    env=self.environment,
                    win=self.depth_refined_win,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='refined depth'))

            else:

                self.depth_refined_win = self.vis.heatmap(
                    depth_refined_t,
                    env=self.environment,
                    opts=dict(
                        xmin=xmin,
                        xmax=xmax,
                        title='refined depth'))

            # Depth map error.
            if self.depth_gt is not None:

                # Detect the valid entries in `self.depth_gt`.
                mask = (self.depth_gt > 0) & (self.depth_gt < float('inf'))

                # Compute the error.
                error = np.abs(self.depth_gt - depth_refined)
                error[~mask] = 0

                # Compute the percentage error.
                percentage_error = depth_percentage_error(
                    depth_refined, self.depth_gt, self.depth_error_threshold)

                # Convert the error to tensor.
                error_t = torch.from_numpy(error).flip([0])

                # Plot the depth map error.
                if self.depth_refined_error_win is not None:

                    self.vis.heatmap(
                        error_t,
                        env=self.environment,
                        win=self.depth_refined_error_win,
                        opts=dict(
                            xmin=0.0,
                            xmax=self.depth_error_threshold,
                            title='refined depth error: {:.2f}% ({})'.format(
                                percentage_error, self.depth_error_threshold)))

                else:

                    self.depth_refined_error_win = self.vis.heatmap(
                        error_t,
                        env=self.environment,
                        opts=dict(
                            xmin=0.0,
                            xmax=self.depth_error_threshold,
                            title='refined depth error: {:.2f}% ({})'.format(
                                percentage_error, self.depth_error_threshold)))

        # ==============================================================================================================

        # Noisy/incomplete normal map.
        if normal is not None:

            if depth is not None:

                # Spatial dimensions.
                height = normal.shape[0]
                width = normal.shape[1]

                # Encode the 3D normals into an RGB image.
                normal_rgb = normal2rgb(normal)

                # Resize the normal map, if too large.
                img_ratio = float(width) / float(height)
                if height > HEIGHT_MAX:
                    normal_rgb = resize_map(
                        normal_rgb, [HEIGHT_MAX, int(HEIGHT_MAX * img_ratio)], order=0)

                # Convert the normal map to tensor.
                normal_rgb_t = torch.from_numpy(np.transpose(normal_rgb, axes=(2, 0, 1)))

                # Plot the normal map.
                if self.normal_win is not None:

                    self.vis.image(
                        normal_rgb_t,
                        env=self.environment,
                        win=self.normal_win,
                        opts=dict(title='input normal'))

                else:

                    self.normal_win = self.vis.image(
                        normal_rgb_t,
                        env=self.environment,
                        opts=dict(title='input normal'))

        # ==============================================================================================================

        # Initial normal map.
        if normal_init is not None:

            if depth_init is not None:

                # Spatial dimensions.
                height = normal_init.shape[0]
                width = normal_init.shape[1]

                # Encode the 3D normals into an RGB image.
                normal_init_rgb = normal2rgb(normal_init)

                # Resize the normal map, if too large.
                img_ratio = float(width) / float(height)
                if height > HEIGHT_MAX:
                    normal_init_rgb = resize_map(
                        normal_init_rgb, [HEIGHT_MAX, int(HEIGHT_MAX * img_ratio)], order=0)

                # Convert the normal map to tensor.
                normal_init_rgb_t = torch.from_numpy(np.transpose(normal_init_rgb, axes=(2, 0, 1)))

                # Plot the normal map.
                if self.normal_init_win is not None:

                    self.vis.image(
                        normal_init_rgb_t,
                        env=self.environment,
                        win=self.normal_init_win,
                        opts=dict(title='initial normal'))

                else:

                    self.normal_init_win = self.vis.image(
                        normal_init_rgb_t,
                        env=self.environment,
                        opts=dict(title='initial normal'))

        # ==============================================================================================================

        # Normal map associated to the refined depth map.
        if normal_refined is not None:

            if depth_refined is not None:

                # Spatial dimensions.
                height = normal_refined.shape[0]
                width = normal_refined.shape[1]

                # Encode the 3D normals into an RGB image.
                normal_refined_rgb = normal2rgb(normal_refined)

                # Resize the normal map, if too large.
                img_ratio = float(width) / float(height)
                if height > HEIGHT_MAX:
                    normal_refined_rgb = resize_map(
                        normal_refined_rgb, [HEIGHT_MAX, int(HEIGHT_MAX * img_ratio)], order=0)

                # Convert the normal map to tensor.
                normal_refined_rgb_t = torch.from_numpy(np.transpose(normal_refined_rgb, axes=(2, 0, 1)))

                # Plot the normal map.
                if self.normal_refined_win is not None:

                    self.vis.image(
                        normal_refined_rgb_t,
                        env=self.environment,
                        win=self.normal_refined_win,
                        opts=dict(title='refined normal'))

                else:

                    self.normal_refined_win = self.vis.image(
                        normal_refined_rgb_t,
                        env=self.environment,
                        opts=dict(title='refined normal'))

        # ==============================================================================================================

        # Depth consistency loss.
        if idepth_consistency_loss is not None:

            if self.idepth_consistency_loss_win is not None:

                self.vis.line(
                    X=idepth_consistency_loss[0],
                    Y=idepth_consistency_loss[1],
                    env=self.environment,
                    win=self.idepth_consistency_loss_win,
                    update='append')

            else:

                self.idepth_consistency_loss_win = self.vis.line(
                    X=idepth_consistency_loss[0],
                    Y=idepth_consistency_loss[1],
                    env=self.environment,
                    opts=dict(
                        xlabel='iterations',
                        ylabel='loss',
                        title='inverse depth consistency loss',
                        markers=True,
                        markersymbol='dot'))

        # ==============================================================================================================

        # Normal consistency loss.
        if inormal_consistency_loss is not None:

            if self.inormal_consistency_loss_win is not None:

                self.vis.line(
                    X=inormal_consistency_loss[0],
                    Y=inormal_consistency_loss[1],
                    env=self.environment,
                    win=self.inormal_consistency_loss_win,
                    update='append')

            else:

                self.inormal_consistency_loss_win = self.vis.line(
                    X=inormal_consistency_loss[0],
                    Y=inormal_consistency_loss[1],
                    env=self.environment,
                    opts=dict(
                        xlabel='iterations',
                        ylabel='loss',
                        title='2D normal consistency loss',
                        markers=True,
                        markersymbol='dot'))

        # ==============================================================================================================

        # Depth regularization loss.
        if regularization_loss is not None:

            if self.regularization_loss_win is not None:

                self.vis.line(
                    X=regularization_loss[0],
                    Y=regularization_loss[1],
                    env=self.environment,
                    win=self.regularization_loss_win,
                    update='append')

            else:

                self.regularization_loss_win = self.vis.line(
                    X=regularization_loss[0],
                    Y=regularization_loss[1],
                    env=self.environment,
                    opts=dict(
                        xlabel='iterations',
                        ylabel='loss',
                        title='regularization loss',
                        markers=True,
                        markersymbol='dot'))

        # ==============================================================================================================

        # Global loss.
        if global_loss is not None:

            if self.global_loss_win is not None:

                self.vis.line(
                    X=global_loss[0],
                    Y=global_loss[1],
                    env=self.environment,
                    win=self.global_loss_win,
                    update='append')

            else:

                self.global_loss_win = self.vis.line(
                    X=global_loss[0],
                    Y=global_loss[1],
                    env=self.environment,
                    opts=dict(
                        xlabel='iterations',
                        ylabel='loss',
                        title='global loss',
                        markers=True,
                        markersymbol='dot'))

    def setup(self,
              env_name: str = None,
              depth_range: Tuple[float, float] = None) -> None:

        # Reset the plot windows.
        self.texture_win = None
        self.depth_win = None
        self.depth_init_win = None
        self.depth_refined_win = None
        self.normal_win = None
        self.normal_init_win = None
        self.normal_refined_win = None
        self.depth_gt_win = None
        self.depth_error_win = None
        self.depth_refined_error_win = None

        # Reset the loss windows.
        self.idepth_consistency_loss_win = None
        self.inormal_consistency_loss_win = None
        self.regularization_loss_win = None
        self.global_loss_win = None

        # Reset the ground truth depth map.
        self.depth_gt = None

        # Reset the plotting depth range.
        self.depth_min = None
        self.depth_max = None

        # Set the new plotting environment name.
        if env_name is not None:
            self.environment = env_name

        # Set the plotting depth range.
        if depth_range is not None:
            self.depth_min = depth_range[0]
            self.depth_max = depth_range[1]
