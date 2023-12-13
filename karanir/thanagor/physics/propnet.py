'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-13 23:13:57
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-13 23:53:39
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

class ParticleEncoder(nn.Module):
    """ a particle encoder that encode the dynamic state and attributes
    use two small encoder layers to perform he encoding
    """
    def __init__(self,input_dim,latent_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # the encoder modules using linear layers and ReLU
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x

class RelationEncoder(nn.Module):
    """ a relation encoder that encode the dynamic state and attributes
    use two small linear encoder layer to encode the relation into propagation effects
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # the encoder modules using linear layers and ReLU
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x

class ParticleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x, res = None):
        """ propagation of relations/particles batch wise
        Args:
            x: [B, N, D]
        Returns:
            x': [B, N, D]
        """
        B, N, D = x.shape
        if self.residual:
            x = x
        else:
            x = self.relu(self.linear(x))
        return x

class PropagationUnits(nn.Module):
    def __init__(self, effect_dim, itrs = 7):
        """ a multi-step propagation unit that propagate the effect across connections in they physics system
        Args:
            effect_dim: the dim of the effect_propgation
            itrs: the number of iteration of effect propagation through the graph, default = 7
        """
        super().__init__()
        self.itrs = itrs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def forward(self, particle_effect, relation_effect = None, edges = None, itrs = None):
        """ propagation according to the following steps:
            1. initalize ht = h0
            2. for i in range(itrs):
            3.   ht = particle_effect 
        """
        if edges is None: edges = 0 # construct the fully connected graph if edges is given
        if itrs is None: itrs = self.itrs

        cumulative_effect = torch.autograd.Variable(torch.zeros())
        cumulative_effect = cumulative_effect.to(self.device)

        # particel encode

class PropNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # consturct the degree of freedom of the physical system
        num_itrs = config.num_itrs # number of iterations of effect propagation
        state_dim = config.state_dim # the dynamic state of freedom [x,v]
        attr_dim = config.attr_dim # the attribute of the states [a], like category, color etc.
        relation_dim = config.relation_dim # the attributs of relations
        effect_dim = config.effect_dim # the effect propagation dim as [effect_dim]
        latent_dim = effect_dim # encoder encode the particle and relation into these dims

        input_dim = state_dim + attr_dim # total input feature dims for single particle.
        hidden_dim = 128 # default hideen dim for encoder and decoders

        # [Encoder Module] for particles and relations
        self.particle_encoder = ParticleEncoder(input_dim, hidden_dim, latent_dim)
        self.relation_encoder = RelationEncoder(input_dim * 2 + relation_dim,hidden_dim, latent_dim)

        # [Decoder Module] for particles [probably also relations but not that sure]
        self.particle_decoder = ParticleDecoder(latent_dim, hidden_dim, input_dim)

        # [PropModule] for effect propagation after the encoder encode the relations and particles
        self.prop_module = PropagationUnits(latent_dim, num_itrs)

    def forward(self, inputs, ground_truth = None):
        """ predict the next state using only the state configuration
        Args:
            inputs: dict contains the following batch wise input
                state: [B,N,Ds]
                attribute: [B,N,Da]
                relation: [B,N,N,Dr]
                edges: SparseTensor[B,N,N]
            ground_truth
        Returns:
            outputs: a dict contains the same key as the next time step prediction
                loss: the loss of prediction is also contained if the ground_truth is not None
        """
        outputs = {}
        if ground_truth == None: train = True
        else: train = False

        # [Input set Up]
        state = inputs["state"]          # [B, N, Ds]
        attributes = inputs["attribute"] # [B, N, Da]
        edges = inputs["edges"]          # [B, N, N ] SparseTensor

        input_features = torch.cat([state, attributes], dim = -1) # [B, N, Ds + Da]

        # [Encoder] for particles and relations
        particle_effects = self.particle_encoder(input_features) # [B, N, D_effect]

        # [Propagation] propagate particles, effects across the graph
        cumulative_effect = self.prop_module(particle_effects, None)

        # [Decoder] for particle (and probably relations)

        # calculate the prediction loss w.r.t the next frame (ground_truth)
        prediction_loss = 0.0
        if train:prediction_loss += 0.0

        outputs["loss"] = prediction_loss
        return outputs