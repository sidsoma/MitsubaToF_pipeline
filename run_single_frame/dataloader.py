import numpy as np
import os
from copy import deepcopy
from math import tan, radians, sqrt, atan, pi, cos, sin
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_extrinsic_matrix(C, p, u):
    # This function turns camera position / look at direction to extrinsic matrix
    # Based on pseudo code from https://ksimek.github.io/2012/08/22/extrinsic/
    # =========== INPUTS =========== #
    # C = origin / camera center
    # p = look at point / target
    # u = up direction 
    # =========== OUTPUTS ========== #
    # camera matrix

    # Calculate the forward vector (the direction we're looking)
    forward = (p - C) / np.linalg.norm(p - C)

    # Calculate the right vector (perpendicular to the forward and up vectors)
    left = np.cross(u, forward)
    left = left / np.linalg.norm(left)

    # Calculate the up vector (perpendicular to the forward and right vectors)
    up = np.cross(forward, left)

    # Construct the rotation matrix
    cam2world = np.hstack([left.reshape(3, 1), up.reshape(3, 1), forward.reshape(3, 1), C.reshape(3, 1)])
    cam2world = np.vstack([cam2world, np.array([[0, 0, 0, 1]])])

    return cam2world

def extract_intrinsic_matrix(x_res, y_res, fov):
    c_x = int(x_res/2) # in pixels
    c_y = int(y_res/2) # in pixels

    # === compute focal length === #
    f_x = x_res / (2*tan(radians(fov/2)))
    f_y = f_x

    intrinsic_matrix = np.array([[f_x,  0 , c_x, 0],
                                 [ 0 , f_y, c_y, 0],
                                 [ 0 ,  0 ,  1 , 0]])

    return intrinsic_matrix

def load_data(cam_x_range, cam_y_range, 
              data_dir_prefix,
              c=3E8, run_mirror_scene=0):
    hists_all = []
    ray_dirs_all = []
    ray_os_all = []
    ray_dirs_hires_all = []

    for cam_x in tqdm(cam_x_range):
        for cam_y in cam_y_range:
            exp_name = f'{data_dir_prefix}_x_{cam_x:.2f}_y_{cam_y:.2f}'

            camera_params = np.load(os.path.join(exp_name, 'params.npz'))
            camera_params = {k: camera_params[k] for k in camera_params.keys()}

            # === Load projector parameters === #
            pixel_coords = camera_params['pixel_coords'] # in homogeneous coordinates
            spot_hw = int(camera_params['spot_hw'])

            # === Load Extrinsic parameters === #
            camera_pos = camera_params['cam_pos']
            look_at = camera_params['look_pos']
            up_dir = camera_params['up_dir']

            # === Load Intrinsic paramers === #
            fov = float(camera_params['fov']) # in degrees
            x_res = int(camera_params['x_res']) # NOTE: CODE WILL ONLY WORK IF X_RES = Y_RES
            y_res = int(camera_params['y_res'])

            # === Load histogram parameters === #
            hists_full = np.load(os.path.join(exp_name, 'output.npz'))
            hists_full = hists_full['I']
            hists_full = hists_full[:, :, 0, :] # only use R channel out of RGB

            numBins = hists_full.shape[-1]
            tMin = camera_params['tMin']
            tMax = camera_params['tMax']
            tRes = camera_params['tRes']

            if run_mirror_scene: 
                num_x_spots = 32
                num_y_spots = 32
            else:
                num_x_spots = camera_params['num_x_spots']
                num_y_spots = camera_params['num_y_spots']
            numSpots = num_x_spots * num_y_spots

            # === Adjust pixel coordinates so that x = right (1, x_res), y = up (1, y_res) === #
            pixel_coords_idx = deepcopy(pixel_coords)
            # y axis is flipped because row indexing goes top --> bottom, pixel goes bottom --> top
            pixel_coords_idx[:, 1] = y_res - pixel_coords_idx[:, 1] 
            # x axis is flipped because column indexing goes right --> left in our coordinate frame
            pixel_coords_idx[:, 0] = x_res - pixel_coords_idx[:, 0]



            # === Increase pixel size by integrating histograms over neighboring pixels === #
            hists = np.zeros((numSpots, numBins))
            hists_test = np.zeros((numSpots, numBins))
            int_proj_image = np.zeros((y_res, x_res))
            tofs = np.argmax(hists_full, axis=-1)
            int_width = 0
            for i in range(numSpots):
                idx_x = int(pixel_coords[i, 0]); idx_y = int(pixel_coords[i, 1])
                # determine range of pixels to integrate over 
                idx_1 = max(idx_y-int_width, 0); idx_2 = min(idx_y+int_width+1, y_res)
                idx_3 = max(idx_x-int_width, 0); idx_4 = min(idx_x+int_width+1, x_res)
                # integrate histogram over neighboring pixels
                transient = np.zeros((numBins, ))
                for y_idx in range(idx_1, idx_2):
                    for x_idx in range(idx_3, idx_4):
                        tof = tofs[y_idx, x_idx]
                        transient[tof] = transient[tof] + hists_full[y_idx, x_idx, tof]
                hists_test[i] = np.sum(hists_full[idx_1:idx_2, idx_3:idx_4], axis=(0, 1))
                hists[i] = transient
                int_proj_image[idx_1:idx_2, idx_3:idx_4] = 1

            # === Extract camera matrices === #
            cam2world = extract_extrinsic_matrix(camera_pos, look_at, up_dir) # (4, 4) matrix ##### DOUBLE CHECK THIS FUNCTION
            intrinsic_matrix = extract_intrinsic_matrix(x_res, y_res, fov) # (3, 3) matrix
            f = intrinsic_matrix[0,0]

            # === Compute ray directions === #
            ray_cam_coords = np.linalg.inv(intrinsic_matrix[:, 0:3]) @ (pixel_coords_idx).T # (3, 3) * (3, numSpots) matrix
            ray_cam_coords_hom = np.vstack([ray_cam_coords, np.ones((1, pixel_coords.shape[0]))]) # (4, numSpots) matrix

            ray_dirs = cam2world @ ray_cam_coords_hom # (4, 144) matrix
            ray_dirs = ray_dirs[0:3, :] / np.linalg.norm(ray_dirs[0:3, :], axis=0) # (3, numSpots) matrix
            ray_dirs = ray_dirs.T # (numSpots, 3) matrix

            # === Add camera offset to the rays to convert them to common world coordinates
            ray_os = np.tile(np.array([[cam_x, cam_y, 0.]]),(numSpots,1))

            # === Create rays for all xres points. Useful for evaluation
            grid_x, grid_y = np.meshgrid(np.arange(0,x_res//4, 1),
                                         np.arange(0,y_res//4, 1))
            pixel_coords_hires = np.stack([grid_x.reshape(-1),
                                           grid_y.reshape(-1),
                                           np.ones((x_res//4)*(y_res//4),)],-1)
            # y axis is flipped because row indexing goes top --> bottom, pixel goes bottom --> top
            pixel_coords_hires[:, 1] = y_res - pixel_coords_hires[:, 1] 
            # x axis is flipped because column indexing goes right --> left in our coordinate frame
            pixel_coords_hires[:, 0] = x_res - pixel_coords_hires[:, 0]
            ray_cam_coords_hires = np.linalg.inv(intrinsic_matrix[:, 0:3]) @ (pixel_coords_hires).T # (3, 3) * (3, numSpots) matrix
            ray_cam_coords_hom_hires = np.vstack([ray_cam_coords_hires, np.ones((1, pixel_coords_hires.shape[0]))]) # (4, numSpots) matrix
            ray_dirs_hires = cam2world @ ray_cam_coords_hom_hires # (4, 144) matrix
            ray_dirs_hires = ray_dirs_hires[0:3, :] / np.linalg.norm(ray_dirs_hires[0:3, :], axis=0) # (3, numSpots) matrix
            ray_dirs_hires = ray_dirs_hires.T # (numSpots, 3) matrix

            # Aggregate all rays and hists        
            ray_dirs_all.append(ray_dirs)
            ray_os_all.append(ray_os)
            hists_all.append(hists)
            ray_dirs_hires_all.append(ray_dirs_hires)

    return ( np.stack(ray_dirs_all,0).reshape(-1,3),
             np.stack(ray_os_all,0).reshape(-1,3),
             np.stack(hists_all,0).reshape(-1, hists.shape[-1]),
             np.stack(ray_dirs_hires_all,0).reshape(-1,3),
             [tMin, tMax, f])