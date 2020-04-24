# depth-refinement-and-normal-estimation

This software is meant to refine a noisy and potentially incomplete depth map,
given the corresponding image.
Since the depth map refinement algorithm underneath this software assumes a piece-wise planar world,
this software estimates a normal map jointly with the refined depth map.  
The software can take advantage of a continuous confidence map with entries in `[0, 1]`,
where `0` denotes unreliable depth values and `1` denotes reliable ones.
In the absence of a confidence map, a pixel is assigned a confidence equal to `1`
if it has a valid depth, `0` otherwise.

This software is released under the MIT license.
If you use this software in your research, please cite the following article:

    @inproceedings{rossi_refinement_2020,
        authors = {Mattia Rossi, Mireille El Gheche, Andreas Kuhn, Pascal Frossard},
        title = {Joint Graph-based Depth Refinement and Normal Estimation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA},
        year = {2020}
    }

## Installation

The software has been tested with Python 3.7 and it has the following dependencies (in brackets are
the tested versions):

- pytorch (v.1.4.0),
- opencv (v.3.4.2),
- visdom (v.1.8.9).

The software relies on `pytorch`, therefore it can run on both CPU and GPU: the latter is recommended.
Processing depth maps of resolution approximately 3000x2000 pixels requires a GPU equipped with a
12 GB memory.
The software does not support the parallel use of multiple GPUs.  
`visdom` is not a mandatory dependency: it is required only for runtime plotting.
In particular, `visdom` permits to observe the progressive refinement of the input depth map from
a web browser, even if the computation is taking place on a remote server.

## How to run the software

The software has a command-line interface, but it can be integrated in a third party code
easily by calling the function `refine` in `refinement.py`.
The following command (new lines must be replaced with spaces) provides and example of usage
of the command-line interface:
    
    python refine
    --image <input_image_path>
    --depth <input_depth_map_path>
    --confidence <input_confidence_map_path>
    --depth_out <refined_depth_map_saving_path>
    --normal_out <estimated_normal_map_saving_path>
    --cam_focal <camera_focal_lenght_x_axis> <camera_focal_lenght_y_axis>
    --cam_center <camera_center_of_projection_x_axis> <camera_center_of_projection_y_axis>
    --depth_min 0.1
    --depth_max 50
    --confidence_threshold 0.5
    --gpu_id 0
    --scale_nb 4
    --lambda_regularization 7.5 7.5 7.5 7.5
    --gamma_regularization 5.5 5.5 5.5 5.5
    --window_size 9 9 9 9
    --patch_size 3 3 3 3
    --sigma_int 0.07 0.07 0.07 0.07
    --sigma_spa 3.0 3.0 3.0 3.0
    --degree_max 20 20 20 20
    --iter_max 4000 3000 2000 1000
    --eps_stop 0.000001 0.000001 0.000001 0.000001
    --attempt_max 50 50 50 50
    --lr_start 0.01 0.01 0.001 0.0001
    --lr_slot_nb 3 3 2 1

The above command performs a refinement of the input depth map adopting a multi-scale refinement
with 4 scales.
As a consequence, the scale-dependent parameters require 4 input values each.
For more details on the software input parameters, please use the help `python ./refine --help`.  
Finally, please note that the input depth and confidence maps must be in binary format (the same used in
[COLMAP](https://github.com/colmap/colmap)).
This is also the same format used to save the refined depth map and the corresponding normal map
on disk.
Reading and writing in binary format is performed by the functions `read_bin_file` and `write_bin_file`,
respectively, in `iofuns.py`.

## Interactive plotting

The software permits to visualize the progress of the input depth map refinement via web browser.
This is implemented using a [VISDOM](https://github.com/facebookresearch/visdom) server.

The VISDOM server can be started with the following command:

    python -m visdom.server -port <visdom_port_number> -base_url /<visdom_base_url> &

where `<visdom_port_number>` and `visdom_base_url` are an arbitrary port and string, respectively.
The server will be accessible at the web page `<server_address>:<visdom_port_number>/<visdom_base_url>`,
where `<server_address>` is the address of the machine where the refinement software runs.
If the software is run locally, then `<server_address>` is `localhost`.

In order to have the software plotting the intermediate results on the VISDOM server, it is necessary
to specify the following two additional parameters when launching the refinement:

    --visdom_display_port <visdom_display_port>
    --visdom_base_url <visdom_base_url>

## License

This software itself is licensed under the MIT license.
The software dependencies and the content of the folder `colmap` may have different licenses:
using these within the depth refinement software may affect the resulting software license.
    
    Copyright (c) 2020,
    ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
    Laboratoire de Traitement des Signaux 4 (LTS4).
    All rights reserved.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    Author: Mattia Rossi (rossi-mattia-at-gmail-com)
