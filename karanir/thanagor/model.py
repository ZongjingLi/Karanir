import torch
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--concept_type",               default = "box")
parser.add_argument("--object_dim",                 default = 100)
parser.add_argument("--concept_dim",                default = 100)
parser.add_argument("--temperature",                default = 0.2)
parser.add_argument("--entries",                    default = 100)
parser.add_argument("--method",                     default = "uniform")
parser.add_argument("--center",                     default = [-0.25,0.25])
parser.add_argument("--offset",                     default = [-0.25,0.25])
parser.add_argument("--domain",                     default = "demo")
config = parser.parse_args(args = [])

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