import torch
import torch.nn as nn

import argparse

from karanir.thanagor.program import *
from karanir.utils import logit

parser = argparse.ArgumentParser()

parser.add_argument("--device",                 default = "cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument("--max_num_masks",         default = 10,       type = int)
parser.add_argument("--backbone_feature_dim",  default = 132)
parser.add_argument("--kq_dim",                default = 64)
parser.add_argument("--channel_dim",           default = 3)

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

class SetNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        latent_dim = 128
        self.fc0 = nn.Linear(config.channel_dim, latent_dim)
        self.fc1 = nn.Linear(latent_dim, config.object_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def forward(self, x, end = None):
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        end = logit(torch.ones(x.shape[1]).to(self.device))
        return end, x

model_dict = {
    "SceneNet":SceneNet,
    "PropNet": PropNet,
    "SetNet": SetNet
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


def evaluate_scenes(dataset, model, verbose = True):
    if isinstance(dataset, list):
        from karanir.utils.data import SimpleDataset
        dataset = SimpleDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    epoch_loss = 0.0
    acc = 0; total = 0
    for sample in loader:
            """Load the scene features and other groundings"""
            features = sample["scene"]["features"]
            percept_end, features = model.perception(features)

            if sample["scene"]["end"] is not None:
                end = sample["scene"]["end"]
                end = logit(end)
                end = torch.min(end, percept_end)
            else:
                end = percept_end

            """Loss calculation for the grounding modules"""
            loss = 0.0
            programs = sample["programs"]
            answers = sample["answers"]

            # perform grounding on each set of scenes
            batch_num = features.shape[0]
            for b in range(batch_num):
                for i,program_batch in enumerate(programs):
                    program = program_batch[b]
                    answer = answers[i][b]
                    """Context for the central executor on the scene"""
                    context = {
                        "end":end[b],
                        "features":features[b],
                        "executor":model.central_executor
                    }
                    q = Program.parse(program)
                    output = q.evaluate({0:context})
                    if answer in ["yes","no"]:
                        if answer == "yes":loss -= torch.log(output["end"].sigmoid())
                        else:loss -= torch.log(1 - output["end"].sigmoid())
                        if output["end"].sigmoid() > 0.5 and answer == "yes": acc += 1
                        if output["end"].sigmoid() < 0.5 and answer == "no": acc += 1
                    else:
                        loss += torch.abs(output["end"] - int(answer))
                        if torch.abs(output["end"] - int(answer)) < 0.5: acc += 1
                    total += 1
                loss /= len(programs) # program wise normalization
            loss /= batch_num # batch wise normalization
            """Clear gradients and perform optimization"""
            epoch_loss += loss.detach().numpy()
    if verbose:print(f"loss:{str(epoch_loss)[:7]} acc:{acc/total}[{acc}/{total}]")
    return acc/total

def fit_scenes(dataset, model, epochs = 1, batch_size = 2, lr = 2e-4, verbose = True):
    import sys
    if isinstance(dataset, list):
        from karanir.utils.data import SimpleDataset
        dataset = SimpleDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    # use Adam optimizer for the optimization of embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sample in loader:
            """Load the scene features and other groundings"""
            features = sample["scene"]["features"]
            percept_end, features = model.perception(features)

            if sample["scene"]["end"] is not None:
                end = sample["scene"]["end"]
                end = logit(end)
                end = torch.min(end, percept_end)
            else:
                end = percept_end

            """Loss calculation for the grounding modules"""
            loss = 0.0
            programs = sample["programs"]
            answers = sample["answers"]

            # perform grounding on each set of scenes
            batch_num = features.shape[0]
            acc = 0; total = 0
            for b in range(batch_num):
                for i,program_batch in enumerate(programs):
                    program = program_batch[b]
                    answer = answers[i][b]
                    """Context for the central executor on the scene"""
                    context = {
                        "end":end[b],
                        "features":features[b],
                        "executor":model.central_executor
                    }
                    q = Program.parse(program)
                    output = q.evaluate({0:context})
                    
                    if answer in ["yes","no"]:
                        if answer == "yes":loss -= torch.log(output["end"].sigmoid())
                        else:loss -= torch.log(1 - output["end"].sigmoid())
                        if output["end"].sigmoid() > 0.5 and answer == "yes": acc += 1
                        if output["end"].sigmoid() < 0.5 and answer == "no": acc += 1
                    else:
                        loss += torch.abs(output["end"] - int(answer))
                        if torch.abs(output["end"] - int(answer)) < 0.5: acc += 1
                    total += 1
                loss /= len(programs) # program wise normalization
            loss /= batch_num # batch wise normalization
            """Clear gradients and perform optimization"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()
            if verbose:
                sys.stdout.write(f"\repoch:{epoch+1} loss:{str(loss)[7:12]} acc:{acc/total}[{acc}/{total}]")
    if verbose:sys.stdout.write("\n")

def generate_scenes(domain, num_scenes):
    import random
    import numpy as np
    max_num_objects = 4
    scenes = []

    """construct the test cases using the domain constraint"""
    type_constraints = domain.type_constraints

    for _ in range(num_scenes):
        single_scene = {}
        """number of object in the scene"""
        end = torch.zeros([max_num_objects])
        num_objs = random.randint(1,max_num_objects)
        end[:num_objs] = 1.0 # set the valid masks for each object

        features = torch.zeros([max_num_objects, len(type_constraints)])

        annotation = [{} for _ in range(max_num_objects)]
        for obj_id in range(max_num_objects):
            for i,ctype in enumerate(type_constraints):
                candidate_values = type_constraints[ctype]
                type_value = np.random.choice(candidate_values)
                type_idx = candidate_values.index(type_value)
                features[obj_id][i] = type_idx

                # add annotations to the scene
                annotation[obj_id][ctype] = type_value

        # add the ground truth annotation
        single_scene["end"] = end
        single_scene["features"] = features
        single_scene["gt"] = annotation
        scenes.append(single_scene)
    return scenes

def generate_grounding(domain, scenes):
    import numpy as np
    type_constraints = domain.type_constraints
    for scene in scenes:
        programs = []
        answers = []
        gt_annotation = scene["gt"]
        # add the Filter Existence type questions
        for ctype in type_constraints:
            category_value = np.random.choice(type_constraints[ctype])
            programs.append("(exists (filter $0 {}))".format(category_value))
            gt_answer = "no"
            for i,obj in enumerate(gt_annotation):
                if scene["end"][i] and obj[ctype] == category_value:
                    gt_answer = "yes"
                    break;
            answers.append(gt_answer)
    
        # add the Filter Count type questions
        for ctype in type_constraints:
            category_value = np.random.choice(type_constraints[ctype])
            programs.append("(count (filter $0 {}))".format(category_value))
            gt_answer = 0
            for i,obj in enumerate(gt_annotation):
                if scene["end"][i] and obj[ctype] == category_value: gt_answer += 1
            answers.append(gt_answer)
    
        scene["programs"] = programs
        scene["answers"] = answers
        scene["scene"] = {"end":scene["end"], "features":scene["features"]}
    return scenes

def render_scene(scenes):
    if not isinstance(scenes, list): scenes = [scenes]
    import matplotlib.pyplot as plt
    fig = plt.figure("render scenes")
    num_scenes = len(scenes)
    return

def helper_scene():
    helper_string = f"""
train scene data bind: diction with following keys
scene: a diction that contains following statements
    end: logits of input scene [N] ;; note about unification, all set in the dataset have the same size, rest use 0 to pad
    features: row features of the input scene [N,D] represent as the 
programs: a list of programs in Lambda Expression
answers: a list of answers for the corresponding programs
    """
    print(helper_string)