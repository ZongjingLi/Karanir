
import torch
from karanir.thanagor.types import *
from karanir.thanagor.program import *

# [Type Specification] of ObjectSet, Attribute, Boolean and other apsects
ObjectSet = baseType("ObjectSet")
Attribute = baseType("Attribute")
Boolean = baseType("Boolean")
Concept = baseType("Concept")
Integer = baseType("Integer")

def expat(tens, idx, num):
    target_shape = [1 for _ in range(1 + len(tens.shape))]
    target_shape[idx] = num
    tens = tens.unsqueeze(idx)
    tens = tens.repeat(target_shape)
    return tens

def logit(x): return torch.log(x / (1 - x))

# [Exist at Least one Element in the Set]
def Exists(x): 
    return {"end":torch.max(x["end"])}
tExists = Primitive("Exists",arrow(ObjectSet, Boolean), Exists)

# [Filter Attribute Concept]
def Filter(objset):
    return lambda concept: {"end":torch.min(objset["end"],objset["executor"].entailment(objset["features"],
            objset["executor"].get_concept_embedding(concept)))}
tFilter = Primitive("Filter",arrow(ObjectSet, Concept, ObjectSet), Filter)

def relate(x,y,z):
    EPS = 1e-6;
    #expand_maks = torch.matmul
    mask = x["executor"].entailment(x["relations"],x["executor"].get_concept_embedding(z))
    N, N = mask.shape
 
    score_expand_mask = torch.min(expat(x["end"],0,N),expat(x["end"],1,N))

    new_logits = torch.min(mask, score_expand_mask)

    return {"end":new_logits}
def Relate(x):
    return lambda y: lambda z: relate(x,y,z)
tRelate = Primitive("Relate",arrow(ObjectSet, ObjectSet, Concept, ObjectSet), Relate)

# [Intersect Sets]{
def Intersect(x): return lambda y: {"end":torch.min(x, y)}
tIntersect = Primitive("Equal",arrow(ObjectSet, ObjectSet, ObjectSet), Intersect)

def Union(x): return lambda y: {"end":torch.max(x["end"], y["end"])}
tUnion = Primitive("Equal",arrow(ObjectSet, ObjectSet, ObjectSet), Union)

# [Do Some Counting]
def Count(x):return {"end":torch.sigmoid(x["end"]).sum(-1)}
tCount = Primitive("Count",arrow(ObjectSet, tint), Count)

def Equal(x):return lambda y:  {"end":8 * (.5 - (x - y).abs())}
tEqual = Primitive("Equal",arrow(treal, treal, Boolean), Equal)