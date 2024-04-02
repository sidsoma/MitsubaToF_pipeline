import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go


def create_ray_visualizer(rays):
    num_rays = rays.shape[0]
    rays_vis = np.zeros((100*num_rays, 4))
    for i in range(num_rays):
        rays_z = np.linspace(0, 3, 100)
        rays_x = rays[i, 0] * rays_z
        rays_y = rays[i, 1] * rays_z
        rays_c = np.vstack([rays_x, rays_y, rays_z])
        # rays_w = np.matmul(R_y, np.matmul(R_x, rays_c)) + x_c
        rays_vis[i*100:(i+1)*100, 0] = i + 1
        rays_vis[i*100:(i+1)*100, 1:] = np.transpose(rays_c)
    rays_vis = pd.DataFrame(data=rays_vis, columns=["ray", "x", "y", "z"])
    return rays_vis

def plot_rays_walls(rays_vis):
    # plot ray and wall
    # lines = px.line_3d(rays_vis, x="x", y="y", z="z", color="ray")
    lines = px.line_3d(rays_vis, x="x", y="y", z="z")
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    camera = dict(eye=dict(x=-1., y=0., z=-2.5), up=dict(x=0, y=1., z=0))
    plot_figure = go.Figure(data=lines, layout=layout)
    # plot_figure.add_traces(data=data)
    plot_figure.update_layout(scene_camera=camera)
    plotly.offline.iplot(plot_figure)

def plot_point_cloud(pt_clouds, cam_pos):

    data = []
    for pts in pt_clouds:
        # Configure the trace.
        plot_points = go.Scatter3d(
            x=np.ndarray.flatten(pts[:, 0]),  # <-- Put your data instead
            y=np.ndarray.flatten(pts[:, 1]),  # <-- Put your data instead
            z=np.ndarray.flatten(pts[:, 2]),  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 2,
                'opacity': 1,
            }
        ) 
        data.append(plot_points)

    origin = go.Scatter3d(
        x=[cam_pos[0]],  # <-- Put your data instead
        y=[cam_pos[1]],  # <-- Put your data instead
        z=[cam_pos[2]],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 2,
            'opacity': 1,
        }
    ) 

    data.append(origin)

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-0.7, y=1.2, z=-2)
    )
    plot_figure.update_layout(scene_camera=camera)
    plotly.offline.iplot(plot_figure)