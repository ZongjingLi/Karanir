'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-14 03:34:00
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-14 03:34:17
 # @ Description: This file is distributed under the MIT license.
'''
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .propagation import GraphPropagatiown
from .competition import Competition

from .backbone import *

from karanir.utils.tensor import gather_tensor, stats_summary, local_to_sparse_global_affinity

def generate_local_indices(img_size, K, padding = 'constant'):
    H, W = img_size
    indice_maps = torch.arange(H * W).reshape([1, 1, H, W]).float()

    # symetric_padding
    assert K % 2 == 1 # assert K is odd
    half_K = int((K - 1) / 2)

    assert padding in ["reflection", "constant"]
    if padding == "constant":
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K)
    
    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size = K, stride = 1)
    local_inds = local_inds.permute(0,2,1)
    return local_inds

def downsample_tensor(x, stride):
    # x should have shape [B, C, H, W]
    if stride == 1:
        return x
    B, C, H, W = x.shape
    x = F.unfold(x, kernel_size=1, stride=stride)  # [B, C, H / stride * W / stride]
    return x.reshape([B, C, int(H / stride), int(W / stride)])

class SceneNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        prop_itrs = 10
        num_masks = 16
        local_window_size = 5
        self.local_window_size = local_window_size
        """visual feature backbone, perform grid convolution"""
        latent_dim = config.backbone_feature_dim
        rdn_args = SimpleNamespace(G0=latent_dim  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True)
        self.backbone = RDN(rdn_args)

        """edge connection encoder, decoder"""
        kq_dim = config.kq_dim
        self.kq_conv = nn.Conv2d(latent_dim, kq_dim, kernel_size = 1, bias = True, padding = "same")
        self.keys_decoder = nn.Linear(kq_dim, kq_dim)
        self.query_decoder = nn.Linear(kq_dim, kq_dim)

        """affinity samples and local indices"""
        self.sample_affinity = True
        supervision_level = 2
        self.supervision_level = supervision_level
        affinity_res = [128,128]
        self.num_affinity_samples = 5 * 5 + 1
        for level in range(supervision_level):
            stride = 2 ** level
            H, W = affinity_res[0] // stride, affinity_res[1] // stride
            buffer_name  = f"local_indices_{H}_{W}"
            local = generate_local_indices([H,W],K = local_window_size)
            self.register_buffer(buffer_name, local)

        """graph and grid propagation methods"""
        self.propagation = GraphPropagation(num_iters = prop_itrs)

        self.compeition = Competition(num_masks = num_masks)

        """normalization of input image and other formats"""

    def forward(self, x, seg = None):
        """
        Args:
            x: single image in the size of BxCxWxH
        Returns:
            the output is a diction bind that contains
        """
        training = 1 if seg is not None else 0
        B, C, H, W = x.shape # BxCxWxH
        # [Backbone features extraction]
        backbone_feature = self.backbone(x) # BxDxWxH

        # [Decode Local Connections]
        kq_features = self.kq_conv(backbone_feature) # BxKxWxH
        kq_features = kq_features.permute(0,2,3,1)
        keys = self.keys_decoder(kq_features).permute(0,3,1,2) # BxQxWxH
        queries = self.query_decoder(kq_features).permute(0,3,1,2) # BxQxWxH

        # [BxHxWxKQ]

        # [Local Connection Computation]
        loss = 0.0
        connections = []
        for level in range(self.supervision_level if training else 1):
            stride = 2 ** level
            # [sample connections]
            if self.sample_affinity:
                sample_inds = self.generate_sample_indices([B, H//stride, W//stride])
            
            # [compute connection strength logits]
            downcast_keys = downsample_tensor(keys , stride)
            downcast_queries = downsample_tensor(queries, stride)
            stats_summary(sample_inds)
            connection_logits = self.compute_logits(
                downcast_keys,
                downcast_queries,
                sample_inds,
            ) * (C ** -0.5)
            stats_summary(connection_logits)
            connections.append(connection_logits)

            if training:loss = self.compute_loss(connection_logits, gt_segments = seg)
        
        # [computer the segmentations]
        masks, agents, alive = self.cpmpute_segment()
        #prop_in = self.propagation()
        #masks, agents, alive, pheno, unharv = self.competition(prop_in)
        masks = 0

        return {
            "loss": loss,
            "connections": connections,
            "masks": masks,
        }

    def compute_segments(self, logits, sample_ind):
        return 0,0,0

    def generate_sample_indices(self,size):
        B, H, W = size
        S = self.num_affinity_samples
        K = self.local_window_size
        # local indices and local masks are stored in the buffers
        local_inds = getattr(self, f"local_indices_{H}_{W}").expand(B,-1,-1)
        device = local_inds.device

        if K ** 2 <= S:
            # sample random global indices with the number of S - K**2
            rand_global_inds = torch.randint(H * W, [B, H * W, S - K**2])
            sample_inds = torch.cat([local_inds, rand_global_inds], -1)
        else:
            sample_inds = local_inds

        sample_inds = sample_inds.reshape([1, B, H*W, S])
        batch_inds = torch.arange(B, device=device).reshape([1, B, 1, 1]).expand(-1, -1, H*W, S)
        node_inds = torch.arange(H*W, device=device).reshape([1, 1, H*W, 1]).expand(-1, B, -1, S)
        sample_inds = torch.cat([batch_inds, node_inds, sample_inds], 0).long()  # [3, B, N, S]
        return sample_inds
    
    def compute_logits(self, key, query, indices):
        B, C, H, W = key.shape
        key = key.reshape([B, C, H * W]).permute(0, 2, 1)      # [B, N, C]
        query = query.reshape([B, C, H * W]).permute(0, 2, 1)  # [B, N, C]

        if self.sample_affinity: # subsample affinity
            gathered_query = gather_tensor(query, indices[[0, 1], ...])
            gathered_key = gather_tensor(key, indices[[0, 2], ...])
            logits = (gathered_query * gathered_key).sum(-1)  # [B, N, K]
        else: # full affinity
            logits = torch.matmul(query, key.permute(0, 2, 1))
        return logits
    
    def compute_loss(self, connections, gt_segments):
        return 0.0