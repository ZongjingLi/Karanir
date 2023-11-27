'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-11-27 21:45:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-11-27 21:46:08
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from karanir.thanagor.program import Program

def wlk_colors():
    blue = torch.tensor([.0, 54.0, 96.0, 256.0]) /256.
    cyan = torch.tensor([0. , 183., 216., 256.]) / 256.
    grey = torch.tensor([58.0, 76.0, 93.0, 256.0]) /256.
    t1 = torch.linspace(0.,0.9,230).unsqueeze(-1)
    t2 = torch.linspace(0.9,1.0,26).unsqueeze(-1)
    lower = (1 - t1) * grey + t1 * blue
    upper =  (1 - t2) * blue + t2 * cyan
    return torch.cat([lower,upper], dim = 0).detach().numpy()

wlk_cmap = ListedColormap(wlk_colors())

def get_wlk_colors(t):
    blue = np.array([.0, 54.0, 96.0, 256.0]) /256.
    cyan = np.array([0. , 183., 216., 256.]) / 256.
    grey = np.array([58.0, 76.0, 93.0, 256.0]) /256.
    t_upper_mask = t >= 0.9
    t_lower_mask = t < 0.9
    lower = (1 - t) * grey + t * blue
    upper =  (1 - t) * blue + t * cyan
    colors = lower * t_lower_mask + upper * t_upper_mask
    return colors

def full_analysis(context):
    concepts = context["executor"].concept_dict
    fig = plt.figure("attribute filters", )

    num_attrs = len(concepts)
    for i,attr in enumerate(concepts):
        ax = fig.add_subplot(1,num_attrs,i + 1)
        outputs = []
        for concept in concepts[attr]:
            p = Program.parse("(Filter $0 {})".format(concept))
            output = p.evaluate({0:context})["end"].unsqueeze(-1).unsqueeze(0).detach()
            outputs.append(output.sigmoid())
        outputs = torch.cat(outputs,dim = 0)

        ax.imshow(get_wlk_colors(outputs))
        plt.yticks(ticks = list(range(len(concepts[attr]))),labels = concepts[attr])
        plt.title(attr)

    relations = context["executor"].relation_dict["relation"]
    fig = plt.figure("relations filters", )

    num_relations = len(relations)
    for i,relation in enumerate(relations):
        ax = fig.add_subplot(1,num_relations,i + 1)

        p = Program.parse("(Relate $0 $0 {})".format(relation))
        output = p.evaluate({0:context})["end"].unsqueeze(-1).detach().sigmoid()

        ax.imshow(get_wlk_colors(output))
        plt.title(relation)
    plt.show()