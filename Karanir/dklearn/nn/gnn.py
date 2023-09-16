import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128):
        super().__init__()
    
    def forward(self, x):
        return x