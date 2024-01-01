'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-24 18:42:10
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-24 18:42:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]