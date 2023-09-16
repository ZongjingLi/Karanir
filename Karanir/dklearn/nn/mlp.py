import torch
import torch.nn as nn

class FCBlock(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim = 132):
        super().__init__()

    def forward(self, x):
        return x