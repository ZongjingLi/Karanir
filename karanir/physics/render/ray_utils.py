# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-11-12 14:49:37
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-12 14:49:39
import numpy as np
import torch

from einops import rearrange

def create_meshgrid(H, W, device):
    Xs = torch.linspace(0, 1, H)
    Ys = torch.linspace(0, 1, W)
    xs, ys = torch.meshgrid([Xs,Ys])
    grid = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1)], dim = -1)
    return grid

def get_ray_directions(H, W, K, flatten = True, return_uv = False, device = "cpu"):
    """
    inputs:
        H, W  image resolution
        K: (3,3) camera instrinsics
        return_uv: whetehr return pixel u,v coordinates
    outputs:
        directiosnL (W,H,3) or (WxH, 3) the direction of the rays in camera coordinate
        uv: (H, W, 2) or (HxW, 2) image coordinates
    """
    grid = create_meshgrid(H,W, device = device) #[H,W,2]
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2] # Get Instrinsic of Camera Matrix
    pass_center = True;
    if pass_center:
        directions = torch.stack([
            (u-cx + 0.5)/fx, (v-cy + 0.5)/fy, torch.ones_like(u)
        ], dim = -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1,2)
    if return_uv: return directions, grid
    return directions

def get_rays(directions, c2w):
    """
    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3,4) or (N,3,4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (N,3) the origin of the rays i the world coordiante
        rays_d: (N,3) the direction of rays in world coordinate
    """
    if len(c2w.shape) == 2:
        rays_d = directions @ c2w[:,:3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')
    rays_o = c2w[..., 3].expand_as(rays_d)

    if(rays_d.shape[1] == 4):
        rays_d = rays_d[:, :3]
        rays_o = rays_o[:, :3]

    return rays_o, rays_d

def create_rays(resolution, focal = 1, origin = False):
    if origin:
        u = torch.linspace(0,1,resolution[0])
        y = torch.linspace(0,1,resolution[1])
    else:
        u = torch.linspace(-1,1,resolution[0])
        v = torch.linspace(-1,1,resolution[1])
    mesh_x, mesh_y = torch.meshgrid(u,v)
    mesh_z = torch.ones_like(mesh_x)
    rays = [x.unsqueeze(-1) for x in [mesh_x, mesh_y, mesh_z]]
    rays = torch.cat(rays, dim = -1)
    return rays