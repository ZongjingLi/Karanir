# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-11 05:13:03
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-12 14:51:01

from karanir.math.homology import TDA

import numpy as np
import torch

import matplotlib.pyplot as plt
from karanir.thanagor.perception.propagation import GraphPropagation
from karanir.thanagor.perception.competition import Competition
from karanir.thanagor.perception.scene_net import SceneNet

comp = Competition(num_masks = 7)

"""Load the dataset to run"""
from karanir.datasets.playroom_dataset import PlayroomDataset

from types import SimpleNamespace

dataset = PlayroomDataset(True, 
SimpleNamespace(precompute_flow=False),
dataset_dir = "/Users/melkor/Documents/datasets/Playroom")

demo_im = torch.zeros([128,128,3])
demo_im[30:60,50:80,0] = 1.0
demo_im[80:100,20:40,1] = 1.0
demo_im += torch.randn_like(demo_im) * 0.001
demo_im = demo_im.clamp(0.0,1.0)
plt.imshow(demo_im)

demo_im = demo_im.unsqueeze(0)
masks, agents, alive, pheno, unharv = comp(demo_im)
print(masks.shape)
print(alive)
print(agents)

agents[:,:,1] = agents[:,:,1]
agents = agents * 64 + 64
for i in range(masks.shape[-1]):
    plt.subplot(1,masks.shape[-1],1 + i)
    plt.scatter(agents[0,i,0],agents[0,i,1])
    plt.imshow(masks[0,:,:,i])
plt.show()