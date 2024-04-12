import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import shutil
import os

from dataloader import load_data
from neural_implicit import CoordinateMLPWithEncoding, generate_voxel_grid
from scipy.io import savemat
import subprocess
import matplotlib.pyplot as plt

data_dir_prefix = 'experiments/sphere_cbox_v6'
cam_x_range = np.linspace(-0.05, 0.05, 3)
cam_y_range = np.linspace(-0.05, 0.05, 3)
c = 3E8
run_mirror_scene = 0
N_select = 5000
num_iter = 10000
log_dir = 'logs/'
log_exp_name = 'exp_07_transmittance_1'
username = 'ad74'

def delete_directory(path_name):
    print(f'Deleting {path_name}')
    shutil.rmtree(path_name) 
 

def visualize_points_with_heatmap(points, heatmap_values, suffix='', folder='/tmp/'):
    """
    Visualize N x 2 points as a 2D scatter plot.
    
    Parameters:
    - points: A Numpy array of shape (N, 2) containing the x and y coordinates of the points.
    - heatmap_values: A Numpy array of shape (N, 1) containing the heatmap values for each point.
    """
    # Ensure heatmap_values is flattened (N,) for compatibility with scatter
    heatmap_values = heatmap_values.flatten()
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=heatmap_values, cmap='jet',marker='.')
    plt.colorbar(scatter, label='Heatmap Value')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('2D Scatter Plot with Heatmap Colors')
    plt.savefig(f'{folder}/ray_dirs_proj_{suffix}.png')
    plt.close()

def evaluate_mlp(mlp, ray_dirs_all, ray_os_all, hists_all, rMin, rMax, f,suffix=''):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, M = hists_all.shape
    criterion = nn.SmoothL1Loss()
    ray_dirs_all = torch.tensor(ray_dirs_all, dtype=torch.float32).to(device)
    ray_os_all = torch.tensor(ray_os_all, dtype=torch.float32).to(device)
    hists_all = torch.tensor(hists_all, dtype=torch.float32).to(device)
        
    # 2. Compute the ray intervals M for all selected rays
    intervals = torch.linspace(rMin, rMax, steps=M+1).to(device) # M+1 to get M intervals
        
    # 3. Uniformly sample a point in each interval and find x, y, z coordinates
    t_values = intervals[:-1] + (torch.rand(N, M, device=device) * (intervals[1:] - intervals[:-1]))
    sampled_points = ray_os_all[:, None, :] + ray_dirs_all[:, None, :] * t_values[..., None]
    sampled_points = sampled_points.reshape(-1, 3)
        
    # 4. Input sampled x,y,z coordinates into MLP
    mlp_output = mlp(sampled_points).squeeze()
    mlp_res = mlp_output.reshape(N, M)
    delta = torch.cat([ 
                       t_values[:,1:]-t_values[:,:-1],
                       torch.zeros_like(t_values[:,[0]])
                       ],-1)
    alphas = 1 - torch.exp(-mlp_res*delta)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alphas[:,[0]]), 1.-alphas +1e-10],-1),-1)[:,:-1]
    final_output = (transmittance**2*mlp_res*delta).view(-1)
    # final_output = mlp_output
        
    # 5. Compute loss
    # loss = criterion(torch.log(final_output+1), torch.log(hists_all.view(-1)+1))
    # loss = criterion(torch.exp(mlp_output), torch.log(hists_all.view(-1)+1))
    loss = criterion(final_output, hists_all.view(-1))
    print(f'Eval Loss: {loss.item()}')
    
    # Compute max along ray to get depth
    hist_max = torch.argmax(hists_all,dim=1)
    hist_r_max = rMin + (rMax - rMin)*hist_max/M
    hist_z_max = ray_os_all[:,2] + ray_dirs_all[:,2]*hist_r_max
    # final_res = torch.exp(torch.exp(final_output.reshape(N,M))) - 1
    final_res = final_output.reshape(N,M)
    final_max = torch.argmax(final_res,dim=1)
    final_r_max = rMin + (rMax - rMin)*final_max/M
    final_z_max = ray_os_all[:,2] + ray_dirs_all[:,2]*final_r_max
    
    # Vizualize depth as max
    # Points projected along the sensor plane
    ray_dirs_proj = ray_os_all[:,:2]+ torch.stack([ray_dirs_all[:,0]/ray_dirs_all[:,2],
                                                   ray_dirs_all[:,1]/ray_dirs_all[:,2]],-1)

    os.makedirs(f'/tmp/{log_exp_name}/', exist_ok = True)
    visualize_points_with_heatmap(ray_dirs_proj.detach().cpu(), hist_z_max.detach().cpu(),
                                  suffix=f'{suffix}_gt',folder=f'/tmp/{log_exp_name}/')
    # visualize_points_with_heatmap(ray_dirs_proj[:32*32*1].detach().cpu(), hist_z_max[:32*32*1].detach().cpu(),
                                #   suffix=f'{suffix}_gt_1',folder=f'/tmp/{log_exp_name}/')
    # visualize_points_with_heatmap(ray_dirs_proj[:32*32*2].detach().cpu(), hist_z_max[:32*32*2].detach().cpu(),
    #                               suffix=f'{suffix}_gt_2',folder=f'/tmp/{log_exp_name}/')
    visualize_points_with_heatmap(ray_dirs_proj.detach().cpu(), final_z_max.detach().cpu(),
                                  suffix=f'{suffix}_rendered',folder=f'/tmp/{log_exp_name}/')
     
    savemat(f"/tmp/{log_exp_name}/eval_output_{suffix}.mat", {"hists_all": hists_all.detach().cpu().numpy(),
                                     "mlp_res": final_res.detach().cpu().numpy(),
                                     "sampled_points":sampled_points.detach().cpu().numpy()})


    return mlp

def copy_results():
    if os.path.exists(os.path.join(log_dir, log_exp_name)):
        delete_directory(os.path.join(log_dir, log_exp_name))
    subprocess.call(f'chmod -R 777 {log_dir}', shell=True)
    subprocess.call(f'mkdir {log_dir}/{log_exp_name}', shell=True)
    subprocess.call(f'su - {username} -c '
                    f'"cd /u/{username}/MitsubaToF_pipeline/run_single_frame && '
                    f' chmod -R 777 {log_dir} && '
                    f'logout "', shell=True)
    subprocess.call(f'cp -r /tmp/{log_exp_name} {log_dir}', shell=True)
    print(f'Files copied to {log_dir}/{log_exp_name}')
def train_mlp(ray_dirs_all, ray_os_all, hists_all, rMin, rMax, f, num_iter, N_select, L=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp = CoordinateMLPWithEncoding(L=L).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    N, M = hists_all.shape
    # For debugging
    N_select = N

    for _ in (range(num_iter)):
        # 1. Randomly select N_select rays
        indices = np.random.choice(N, N_select, replace=False)
        selected_dirs = torch.tensor(ray_dirs_all[indices], dtype=torch.float32).to(device)
        selected_os = torch.tensor(ray_os_all[indices], dtype=torch.float32).to(device)
        selected_hists = torch.tensor(hists_all[indices], dtype=torch.float32).to(device)
        
        # 2. Compute the ray intervals M for all selected rays
        intervals = torch.linspace(rMin, rMax, steps=M+1).to(device) # M+1 to get M intervals
        
        # 3. Uniformly sample a point in each interval and find x, y, z coordinates
        t_values = intervals[:-1] + (torch.rand(N_select, M, device=device) * (intervals[1:] - intervals[:-1]))
        sampled_points = selected_os[:, None, :] + selected_dirs[:, None, :] * t_values[..., None]
        sampled_points = sampled_points.reshape(-1, 3)
        
        # 4. Input sampled x,y,z coordinates into MLP
        mlp_output = mlp(sampled_points).squeeze()
        mlp_res = mlp_output.reshape(N_select, M)
        delta = torch.cat([ 
                           t_values[:,1:]-t_values[:,:-1],
                           1e10*torch.ones_like(t_values[:,[0]])
                           ],-1)
        alphas = 1 - torch.exp(-mlp_res*delta)
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alphas[:,[0]]), 1.-alphas +1e-10],-1),-1)[:,:-1]
        final_output = (transmittance**2*mlp_res*delta).view(-1)
        # final_output = mlp_output
        
        # 5. Compute loss
        # loss = criterion(torch.log(final_output+1), torch.log(selected_hists.view(-1)+1))
        loss = criterion(final_output, selected_hists.view(-1))
        
        # 6. Update MLP parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if _ % 100 == 0:
            print(f'Iteration {_+1}, Loss: {loss.item()}')

        if _ % 1000 == 0:
            evaluate_mlp(mlp, ray_dirs_all, ray_os_all, hists_all, 
                        rMin, rMax, f, suffix=f'coarse_{_:06d}')
            x_res = 512 
            with torch.no_grad():
                mlp = evaluate_mlp(mlp, ray_dirs_hires_all[:(x_res//4)**2], ray_os_all[[0]], np.zeros(((x_res//4)**2, hists_all.shape[-1])), 
                            rMin, rMax, f, suffix=f'fine_{_:06d}')
            copy_results()
    return mlp


(ray_dirs_all, ray_os_all, hists_all, ray_dirs_hires_all, [tMin, tMax, f]) = load_data(cam_x_range, 
                                                                      cam_y_range,
                                                                      data_dir_prefix,
                                                                      c, run_mirror_scene)

# Distance along the ray corresponds to half of travel time
rMin, rMax = tMin/2, tMax/2

mlp = train_mlp(ray_dirs_all, ray_os_all, hists_all, 
                                    rMin, rMax, f, num_iter, N_select)

def evaluate_voxel_grid(mlp, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    """
    Evaluate the implicit representation across a voxel grid and save the output as a .mat file.
    """
    device = next(mlp.parameters()).device
    voxel_grid = generate_voxel_grid(x_min, x_max, y_min, y_max, z_min, z_max, resolution).to(device)
    voxel_grid_flat = voxel_grid.reshape(-1, 3)  # Flatten the grid for batch processing

    # Predict the implicit representation values for the voxel grid
    with torch.no_grad():
        predictions = mlp(voxel_grid_flat).cpu().numpy()

    predictions = predictions.reshape(resolution, resolution, resolution)

    # Save the voxel grid predictions to a .mat file
    savemat("/tmp/voxel_grid_output.mat", {"voxel_grid": predictions})
    subprocess.call("cp /tmp/voxel_grid_output.mat . ", shell=True)


# evaluate_voxel_grid(mlp, x_min=-0.6, x_max=0.6, y_min=-0.6, y_max=0.6, z_min=0.2, z_max=1.4, resolution=201)

