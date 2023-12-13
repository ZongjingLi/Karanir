'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-14 03:34:00
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-14 03:34:17
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

class SceneNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x