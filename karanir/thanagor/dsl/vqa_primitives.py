'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-11-28 04:10:30
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-26 23:14:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
from karanir.thanagor.types import *
from karanir.thanagor.program import *

# [Type Specification] of ObjectSet, Attribute, Boolean and other apsects
ObjectSet = baseType("ObjectSet")
Attribute = baseType("Attribute")
Boolean = baseType("Boolean")
Concept = baseType("Concept")
Integer = baseType("Integer")
CodeBlock = baseType("CodeBlock")

from karanir.utils.tensor import logit

def expat(tens, idx, num):
    target_shape = [1 for _ in range(1 + len(tens.shape))]
    target_shape[idx] = num
    tens = tens.unsqueeze(idx)
    tens = tens.repeat(target_shape)
    return tens

# [Exist at Least one Element in the Set]
def Exists(x): 
    return {"end":torch.max(x["end"]), "executor":x["executor"]}
tExists = Primitive("exists",arrow(ObjectSet, Boolean), Exists)

# [Filter Attribute Concept]
def _TypeFilter(objset,concept,executor):
    filter_logits = torch.zeros_like(objset["end"])
    parent_type = executor.get_type(concept)
    for candidate in executor.type_constraints[parent_type]:
        filter_logits += executor.entailment(objset["features"],
            executor.get_concept_embedding(candidate)).sigmoid()

    div = executor.entailment(objset["features"],
            executor.get_concept_embedding(concept)).sigmoid()
    filter_logits = logit(div / filter_logits)
    return{"end":torch.min(objset["end"],filter_logits), "executor":objset["executor"]}
    
def Filter(objset):
    return lambda concept: _TypeFilter(objset, concept, objset["executor"])
tFilter = Primitive("filter",arrow(ObjectSet, Concept, ObjectSet), Filter)

def relate(x,y,z):
    EPS = 1e-6;
    #expand_maks = torch.matmul
    mask = x["executor"].entailment(x["relations"],x["executor"].get_concept_embedding(z))
    N, N = mask.shape
 
    score_expand_mask = torch.min(expat(x["end"],0,N),expat(x["end"],1,N))

    new_logits = torch.min(mask, score_expand_mask)

    return {"end":new_logits, "executor":x["executor"]}
def Relate(x):
    return lambda y: lambda z: relate(x,y,z)
tRelate = Primitive("relate",arrow(ObjectSet, ObjectSet, Concept, ObjectSet), Relate)

# [Intersect Sets]{
def Intersect(x): return lambda y: {"end":torch.min(x, y)}
tIntersect = Primitive("intersect",arrow(ObjectSet, ObjectSet, ObjectSet), Intersect)

def Union(x): return lambda y: {"end":torch.max(x["end"], y["end"])}
tUnion = Primitive("equal",arrow(ObjectSet, ObjectSet, ObjectSet), Union)

# [Do Some Counting]
def Count(x):return {"end":torch.sigmoid(x["end"]).sum(-1), "executor":x["executor"]}
tCount = Primitive("count",arrow(ObjectSet, tint), Count)

def Equal(x):return lambda y:  {"end":8 * (.5 - (x - y).abs()), "executor":x["executor"]}
tEqual = Primitive("equal",arrow(treal, treal, Boolean), Equal)

def If(x,y): x["executor"].execute(y,x["end"])

tIf = Primitive("if", arrow(Boolean, CodeBlock), If)

def Assign(x, y):return {"end1":x["end"], "end2":y["end"]}
tAssign = Primitive("assign", arrow(Attribute, Attribute), Assign)

def Forall(condition,set): return 
tForall = Primitive("forall", Boolean, Forall)

def And(x): return lambda y: {"end":torch.min(x["end"],y["end"])}
tAnd = Primitive("and", arrow(Boolean, Boolean, Boolean), And)

def Or(x): return lambda y: {"end": torch.max(x["end"],y["end"])}
tOr = Primitive("or", arrow(Boolean, Boolean, Boolean), Or)

tTrue = Primitive("true",Boolean,{"end":logit(torch.tensor(1.0))})
tFalse = Primitive("false",Boolean,{"end":logit(torch.tensor(0.0))})