import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        # x is expected to have shape [N, 3] for (x, y, z) coordinates
        frequencies = 2.0 ** torch.arange(self.L, dtype=torch.float32).to(x.device)
        encodings = [fn(x[..., None] * frequencies.view((1, 1, -1))).reshape(x.shape[0], -1) for fn in (torch.sin, torch.cos)]
        return torch.cat(encodings, dim=-1)

class CoordinateMLPWithEncoding(nn.Module):
    def __init__(self, L=10):
        super(CoordinateMLPWithEncoding, self).__init__()
        self.encoding = SinusoidalPositionalEncoding(L=L)
        input_dim = 6 * L  # Encoded dimension
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjusted input layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.act = lambda x : torch.exp(torch.exp(x)) - 1
        # self.act = lambda x : x
        self.act = lambda x : torch.sigmoid(x)
    
    def forward(self, x):
        encoded_x = self.encoding(x)
        return self.act(self.layers(encoded_x))

def generate_voxel_grid(x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    """
    Generate a voxel grid within the specified bounds with a given resolution.
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    voxel_grid = np.stack([xv, yv, zv], axis=-1)
    return torch.tensor(voxel_grid, dtype=torch.float32)