from karanir.thanagor.types import *
from karanir.thanagor.program import Program

from karanir.datasets.playroom_dataset import PlayroomDataset

from torch.utils.data import Dataset, DataLoader

dataset_dir = "/Users/melkor/Documents/datasets/Playroom"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compute_flow",           default = False)
parser.add_argument("--precompute_flow",        default = False)
args = parser.parse_args(args = [])

dataset = PlayroomDataset(True, args, dataset_dir = dataset_dir, flow = False)
loader = DataLoader(dataset, batch_size = 2, shuffle = True)

for sample in loader:
    break;

import matplotlib.pyplot as plt

def visualize_sample(sample, fig_name = "visualize sample"):
    fig = plt.figure(fig_name)
    for i,k in enumerate(sample):
        ax = fig.add_subplot(1,3,i+1)
        ax.imshow(sample[k][0])
    plt.show()

from karanir.algs.search import run_heuristic_search
from karanir.algs.search.configuration_space import ProblemSpace

import karanir.thanagor.domain as dom
from karanir.thanagor.dsl.vqa_primitives import *

from lark import Lark, Tree, Transformer, v_args

with open("ankaranir.icstruct") as struct_file:
    domain_string = ""
    for line in struct_file: domain_string += line

domain_string = r"""(define (domain blocks-world)
    (:types
        block - object
        id - int64
        color - object
    )

    (:predicates
        (color ?x - block)
        (is-red ?x - color)
        (clear ?x - block)          ;; no block is on x
        (on ?x - block ?y - block)  ;; x is on y
        (robot-holding ?x - block)  ;; the robot is holding x
        (robot-handfree)            ;; the robot is not holding anything
    )
)"""

#print(domain_string)
domain = dom.load_domain_string(domain_string)
domain.print_summary()

from karanir.thanagor.knowledge import CentralExecutor
from karanir.thanagor import config

config.concept_type = "cone"
executor = CentralExecutor(domain, config)

context = {
    "end": logit(torch.ones(7)),
    "features": torch.randn([7,100]),
    "executor": executor
    }

red = Primitive("red", Concept, "red")
p = Program.parse("(Count (Filter $0 red))")

print(p.evaluate({0:context})["end"])