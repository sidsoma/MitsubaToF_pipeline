import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
import time
from argparse import ArgumentParser
plt.ioff()
import sys
sys.path.append('.')


def run_scan_fn(cam_x, cam_y):

    # ============================================================== #
    # ===================== DEFINE PARAMETERS ====================== #
    # ============================================================== #

    run_diffuse_test = True

    # =================== Define Experiment Name =================== #
    if run_diffuse_test:
        # exp_name = f'diffuse_patch_test_2_x_{cam_x:.2f}_y_{cam_y:.2f}'
        # scene_name = '../scenes/test_patch_2.xml'
        # numSamples = 16
        exp_name = f'sphere_cbox_v6_x_{cam_x:.2f}_y_{cam_y:.2f}'
        scene_name = '../scenes/sphere_cbox.xml'
        numSamples = 16
    else:
        exp_name = 'mirror_test_32x32'
        scene_name = '../scenes/mirror.xml'
        numSamples = 3200

    save_dir = 'experiments'
    exr_script = 'exr2mat.py'
    username = 'ad74'

    # ============ Define Intrinsic Parameters (Fixed) ============= #
    # Field of view 
    # fov = 36
    fov = 90
    fovAxis = 'x'
    # Camera pixel resolution
    x_res = 512
    y_res = 512
    # Number of laser Spots
    num_x_spots = 32
    num_y_spots = 32
    # Laser spot half-width in pixels (spot_hw * 2 = spot width in pixels)
    spot_hw = 0 

    # ================= Define Extrinsic Parameters ================= #
    # Camera Location (world coordinates)
    cam_x = cam_x
    cam_y = cam_y
    cam_z = 0
    # Camera Lookat position (world coordinates)
    look_x = cam_x
    look_y = cam_y
    # look_y = -1
    look_z = 1

    # === Simulator Parameters === #
    # tMin = 2.2 # units in pathlength
    # tMin = 1.8
    # tMax = 4.8 # units in pathlength
    # # tRes = 0.001 # units in pathlength
    # tRes = 0.01
    tMin = 1.4
    tMax = 2.8 # units in pathlength
    # tRes = 0.001 # units in pathlength
    tRes = 0.01

    # ============================================================== #
    # ============== DO NOT MODIFY BELOW THIS POINT ================ #
    # ============================================================== #
    def delete_directory(path_name):
        print(f'Deleting {path_name}')
        shutil.rmtree(path_name) 

    # === Create experiment directory === #
    # output_dir = os.path.join(save_dir, exp_name)
    # # if os.path.exists(exp_name):
    # #     delete_directory(exp_name)
    # # os.mkdir(os.path.join(save_dir, exp_name))
    # if os.path.exists(output_dir): 
    #     delete_directory(output_dir)
    # os.mkdir(output_dir)
    if os.path.exists('/mitsuba/' + exp_name): 
        delete_directory('/mitsuba/' + exp_name)
    os.mkdir('/mitsuba/' + exp_name)

    # === Compute projector image === #
    x_locs = np.linspace(spot_hw + 5, x_res-5-spot_hw, num_x_spots)
    y_locs = np.linspace(spot_hw + 5, y_res-5-spot_hw, num_y_spots)
    grid_x, grid_y = np.meshgrid(x_locs, y_locs, indexing='xy')

    pixel_coords = np.zeros((num_x_spots*num_y_spots, 3))
    proj_img = np.zeros((y_res, x_res))
    i = 0
    for ix in range(num_x_spots):
        for iy in range(num_y_spots):
            idx_y = int(grid_y[iy, ix])
            idx_x = int(grid_x[iy, ix])
            pixel_coords[i] = np.array([idx_x, idx_y, 1])
            proj_img[idx_y-spot_hw:idx_y+spot_hw+1, idx_x-spot_hw:idx_x+spot_hw+1] = 1
            i += 1
    # For white projector image generation
    # proj_img = np.ones((y_res, x_res))
    # proj_img[0,0] = 0

    print("Saving projector image and camera parameters...")
    cam_pos = np.array([cam_x, cam_y, cam_z])
    look_pos = np.array([look_x, look_y, look_z])
    up_dir = np.array([0, 1, 0])
    camera_params = {'pixel_coords': pixel_coords, 'proj_img': proj_img, 'spot_hw': spot_hw,
                    'cam_pos': cam_pos, 'look_pos': look_pos, 'up_dir': up_dir, 'fov': fov,
                    'tMin': tMin, 'tMax': tMax, 'tRes': tRes, 'x_res': x_res, 'y_res': y_res,
                    'num_x_spots': num_x_spots, 'num_y_spots': num_y_spots}
    np.savez(f'/mitsuba/{exp_name}/params.npz', **camera_params)

    plt.figure(figsize=(12, 12), frameon=False)
    plt.imshow(proj_img, interpolation='none', cmap='gray')
    plt.axis('off')
    proj_img_dir = os.path.join('/mitsuba/', exp_name, 'proj_img.png')
    plt.savefig(proj_img_dir, bbox_inches='tight', pad_inches=0)

    # === Run Mitsuba on defined scene and camera parameters === #
    # output_file = os.path.join(output_dir, 'output.exr')
    output_file = os.path.join('/mitsuba/', exp_name, 'output.exr')
    print("Running simulation...")
    start_time = time.time()
    subprocess.call(f'mitsuba camera.xml ' # .xml file for all experiments
                    f'-D fov={fov} -D fovAxis={fovAxis} ' 
                    f'-D cam_x={cam_x} -D cam_y={cam_y} -D cam_z={cam_z} ' 
                    f'-D look_x={look_x} -D look_y={look_y} -D look_z={look_z} ' 
                    f'-D numSamples={numSamples} '
                    f'-D x_res={x_res} -D y_res={y_res} '
                    f'-D tMin={tMin} -D tMax={tMax} -D tRes={tRes} '
                    f'-D proj_img={proj_img_dir} '
                    f'-D scene={scene_name} '
                    f'-o {output_file} ' # file and directory to save to
                    f'-q -j 10', # quiet mode, simultaneously schedule several scenes
                    shell=True)

    end_time = time.time()
    print("Elapsed Time: ", end_time - start_time, " seconds")

    # === Extract data from exr files === #
    print("Converting exr to npy...")
    # subprocess.call(f"python {exr_script} {output_file} {output_dir}/output", shell=True)
    subprocess.call(f"python {exr_script} {output_file} /mitsuba/{exp_name}/output", shell=True)


    # === Extract data from exr files === #
    if os.path.exists(os.path.join(save_dir, exp_name)):
        delete_directory(os.path.join(save_dir, exp_name))
    subprocess.call(f'chmod -R 777 {save_dir}', shell=True)
    subprocess.call(f'mkdir {save_dir}/{exp_name}', shell=True)
    subprocess.call(f'su - {username} -c '
                    f'"cd /u/{username}/MitsubaToF_pipeline/run_single_frame && '
                    f' chmod -R 777 {save_dir} && '
                    f'logout "', shell=True)
    # subprocess.call(f'cd /u/{username}/MitsubaToF_pipeline/run_single_frame', shell=True)
    # subprocess.call(f'chmod -R 777 {save_dir}', shell=True)
    # subprocess.call('logout', shell=True)
    # os.system('pkill -KILL -u sidsoma')
    subprocess.call(f'cp -r /mitsuba/{exp_name} {save_dir}', shell=True)
    print(f'Files copied to {save_dir}/{exp_name}')

num_x = 3
num_y = 3
cam_x_range = np.linspace(-0.05,0.05, num_x)
cam_y_range = np.linspace(-0.05,0.05, num_y)

for cam_x in cam_x_range:
    for cam_y in cam_y_range:
        run_scan_fn(cam_x, cam_y)