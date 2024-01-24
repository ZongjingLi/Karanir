# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-11 05:13:03
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-12 14:51:01
import numpy as np
import torch

import matplotlib.pyplot as plt
from karanir.thanagor.model import config
from karanir.thanagor.perception.propagation import GraphPropagation
from karanir.thanagor.perception.competition import Competition
from karanir.thanagor.perception.scene_net import SceneNet
from karanir.datasets.sprites_dataset import SpriteDataset, DataLoader
from karanir.utils.tensor import gather_loss, stats_summary
from karanir.logger import set_output_file, get_logger
import sys

set_output_file("logs/demo_perceptId64.txt")
logger = get_logger("PerceptIdDemo64")

epochs = 10000
batch_size = 1
lr = 2e-4
train_dataset = SpriteDataset("train")
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

config.channel_dim = 4
config.resolution = (64,64)
model = SceneNet(config)
logger.critical("SceneNet model is built succesfully.")
pth_name = "model1.pth"
#pth_name = None
if pth_name is not None:
    try:
        model.load_state_dict(torch.load(pth_name))
        logger.critical("SceneNet load parameters from :{}".format(pth_name))
    except:
        logger.exception("Failed to load the state dict from {}".format(pth_name))
for sample in train_loader:
    #break
    ims = sample["img"]
    masks = sample["masks"]
    plt.figure("origin");plt.cla()
    plt.imshow(ims[0].permute(1,2,0))
    outputs = model(ims, masks.long().unsqueeze(1))
    masks = outputs["masks"]
    fig = plt.figure("masks")
    for i in range(masks.shape[-1]):
        ax = fig.add_subplot(2,5,1+i)
        ax.imshow(masks[0,:,:,i])
    print(gather_loss(outputs)["loss"].detach())
    print(outputs["alive"].flatten())
    plt.show()

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

logger.critical("start the training experiment on sprite_env.")
for epoch in range(epochs):
    epoch_loss = 0.0
    itr = 0
    for sample in train_loader:
        itr += 1
        ims = sample["img"]
        masks = sample["masks"]
        outputs = model(ims, masks.long().unsqueeze(1))

        working_loss = gather_loss(outputs)["loss"]

        optimizer.zero_grad()
        working_loss.backward()
        optimizer.step()
        epoch_loss += working_loss.detach()
        sys.stdout.write(f"\repoch:{epoch+1} itr:{itr} loss:{working_loss}")
    logger.critical(f"epoch:{epoch+1} loss:{epoch_loss}")
    torch.save(model.state_dict(),f"{pth_name}")