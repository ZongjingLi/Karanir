import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .embedding  import build_box_registry
from .entailment import build_entailment
from karanir.utils import freeze
from karanir.utils.misc import *
from karanir.thanagor.types import baseType

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

class SceneGraphRepresentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.effective_level = 1
        self.max_level = 4

    @property
    def top_objects(self):
        return 0

def expat(tens, idx, num):
    target_shape = [1 for _ in range(1 + len(tens.shape))]
    target_shape[idx] = num
    tens = tens.unsqueeze(idx)
    tens = tens.repeat(target_shape)
    return tens

class CentralExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, domain, config):
        super().__init__()
        BIG_NUMBER = 100

        self.entailment = build_entailment(config)
        self.concept_registry = build_box_registry(config)

        # [Types]
        self.types = domain.types
        for type_name in domain.types:
            baseType(type_name)

        # [Predicates]
        self.predicates = domain.predicates

        # [Actions]
        self.actions = domain.actions

        # [Word Vocab]
        self.concept_dict = {
            "color":["red","green","blue"],
            "shape":["circle","square","diamond"]
            }
        self.relation_dict = {
            "relation":["left","right"]
        }
        concept_vocab = [c for a in self.concept_dict for c in self.concept_dict[a]]

        relation_vocab = [c for a in self.relation_dict for c in self.relation_dict[a]]
        self.relation_encoder = nn.Linear(config.object_dim * 2, config.object_dim)

        self.concept_vocab = concept_vocab
        self.concept_vocab.extend(relation_vocab)

        # args during the execution
        self.kwargs = None 

        self.effective_level = BIG_NUMBER
    
    def full_analysis(self, scene):
        return
    
    def build_relations(self, scene):
        end = scene["end"]
        features = scene["features"]
        N, D = features.shape
        cat_features = torch.cat([expat(features,0,N),expat(features,1,N)], dim = -1)
        relations = self.relation_encoder(cat_features)
        return relations
    
    def spectrum(self,node_features, concepts = None):
        masks = []
        if concepts is None: concepts = self.concept_vocab
        for concept in concepts: 
            masks.append(self.concept_registry(\
                node_features, \
                self.get_concept_embedding(concept))
                )
        return masks

    def entail_prob(self, features, concept):
        kwargs = {"end":[torch.ones(features.shape[0])],
             "features":[features]}
        q = self.parse("filter(scene(),{})".format(concept))
        o = self(q, **kwargs)
        return o["end"]
    
    def all_embeddings(self):
        return self.concept_vocab, [self.get_concept_embedding(emb) for emb in self.concept_vocab]

    def get_concept_embedding(self,concept):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        concept_index = self.concept_vocab.index(concept)
        idx = torch.tensor(concept_index).unsqueeze(0).to(device)
        return self.concept_registry(idx)

    def forward(self, q, **kwargs):
        self.kwargs = kwargs
        return q(self)


class MetaLearner(nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = MetaLearner.get_name(cls.__name__)
        CentralExecutor.NETWORK_REGISTRY[name] = cls
        cls.name = name

    @staticmethod
    def get_name(name):
        return name[:-len('Learner')]

    def forward(self, p):
        return {}

    def compute_logits(self, p, **kwargs):
        return p.evaluate_logits(self, **kwargs)


class PipelineLearner(nn.Module):
    def __init__(self, network, entailment):
        super().__init__()
        self.network = network
        self.entailment = entailment

    def forward(self, p):
        shots = []
        for q in p.train_program:
            end = q(self)["end"]
            index = end.squeeze(0).max(0).indices
            shots.append(q.object_collections[index])
        shots = torch.stack(shots)
        if not p.is_fewshot:
            shots = shots[0:0]
        fewshot = p.to_fewshot(shots)
        return fewshot(self)

    def compute_logits(self, p, **kwargs):
        return self.network.compute_logits(p, **kwargs)