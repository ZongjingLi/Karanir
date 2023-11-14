import torch
import torch.nn as nn

class KaranirThanagor(nn.Module):
    def __init__(self, domain = None):
        super().__init__()
        self.domain = domain

        # [Knowledge Construct]
        self.knowledge = None

        # [Planning, Reasoning Constuct]
        self.planer = None

    def fit(self,data, task = None):
        return 