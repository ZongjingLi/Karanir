import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128):
        super().__init__()
        self.pre_map = nn.Linear(input_dim, latent_dim)
        self.final_map = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x, adj, residual = True):
        """
        inputs:
            x: [B,N,D], adj: [B,N,N]
        outputs:
            y: [B,N,E]
        """
        x = self.pre_map(x)
        if isinstance(adj, torch.Sparse):
            pass
        else:
            if residual: x = x + torch.bmm(adj, x)
            x = torch.bmm(adj, x)
        x = self.final_map(x)
        return x
    
class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128, 
                    with_v = False, retract = True, normalize = True):
        super().__init__()
        self.k_map = nn.Linear(input_dim, latent_dim)
        self.q_map = nn.Linear(input_dim, latent_dim)

        self.v_map = nn.Linear(input_dim, output_dim) if with_v else nn.Identity()


        self.retract = retract
        self.normalize = normalize

    def forward(self, x, adj):
        """
        inputs:x: [B,N,D], adj: [B,N,N]
        outputs:y: [B,N,E]
        """
        if self.normlaize: x = nn.functional(x, p = 2)
        s = math.sqrt(x.shape[-1])
        ks = self.k_map(x) # BxNxDk
        qs = self.q_map(x) # BxNxDq
        attn = torch.bmm(ks/s,qs.transpose(1,2)/s)
        if self.retract: attn -= 0.5
        attn = attn * adj

        vs = self.v_map(x) # BxNxDv
        outputs = torch.bmm(attn, vs)
        if self.normalize: outputs = nn.functional.normalize(outputs, p = 2)
        return outputs