import torch
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()

""" [Concept Model]"""
parser.add_argument("--concept_type",          default = "cone")
parser.add_argument("--object_dim",            default = 100)
parser.add_argument("--concept_dim",           default = 100)
parser.add_argument("--temperature",           default = 0.2)
parser.add_argument("--entries",               default = 100)
parser.add_argument("--method",                default = "uniform")
parser.add_argument("--center",                default = [-0.25,0.25])
parser.add_argument("--offset",                default = [-0.25,0.25])
parser.add_argument("--domain",                default = "demo")

"""[Perception Model]"""
parser.add_argument("--perception_model_name", default = "SceneNet")
parser.add_argument("--resolution",            default = (128,128))

""" [Physics Model]"""
parser.add_argument("--physics_model_name",    default = "PropNet")
parser.add_argument("--state_dim",             default = 2 + 2,    type = int,     help = "the dynamic state dim, normally it's the x + v")
parser.add_argument("--attr_dim",              default = 5,        type = int,     help = "the number of attributes for each particle")
parser.add_argument("--relation_dim",          default = 2,        type = int,     help = "the number of relational features between particles")
parser.add_argument("--effect_dim",            default = 32,       type = int,     help = "the effect propagation dim")
parser.add_argument("--num_itrs",              default = 7,        type = int)

config = parser.parse_args(args = [])

from .perception import SceneNet
from .physics import PropNet
from .knowledge import CentralExecutor

model_dict = {
    "SceneNet":SceneNet,
    "PropNet": PropNet,
}

class KaranirThanagor(nn.Module):
    def __init__(self, domain, config):
        super().__init__()
        self.config = config
        self.domain = domain

        # [Perception Model]
        self.resolution = config.resolution
        self.perception = model_dict[config.perception_model_name](config)

        # [Physics Model]
        self.evolutor = model_dict[config.physics_model_name](config)

        # [Central Knowledge Executor]
        self.central_executor = CentralExecutor(domain, config)

        # [Neuro Implementations]
        self.implementations = nn.ModuleDict()

    def fit(self,data, task = None):
        return 

    def print_summary(self):
        summary_string = f"""
[Perception Model]
perception:{self.config.perception_model_name}  ;; the name of the perception model
resolution:{self.resolution}  ;; working resolution of the object centric perception

[Physics Model]
evolutor: {self.config.physics_model_name}  ;; the name of the evolution model
state_dim: {self.config.state_dim}  ;; dynamic state dim for each particle (position and momentum)
attr_dim: {self.config.attr_dim}    ;; attribute dim for each particle state
relation_dim: {self.config.relation_dim}    ;; the number of relations between objects

[Central Knowlege Base]
concept_type: {self.config.concept_type}    ;; the type of the concept structure
concept_dim: {self.config.concept_dim}      ;; the dim of the concept embedding
object_dim: {self.config.object_dim}        ;; the dim of the object space embedding
"""
        print(summary_string)
        if self.domain is not None:self.domain.print_summary()