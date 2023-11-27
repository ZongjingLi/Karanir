from webbrowser import get
from karanir.thanagor import *
from karanir.thanagor.knowledge.executor import CentralExecutor

KFT = KaranirThanagor("The Domain")

from karanir.thanagor.types import *
from karanir.thanagor.program import *

import sys

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
    mask = executor.entailment(x["relations"],x["executor"].get_concept_embedding(z))
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

def Union(x): return lambda y: {"end":torch.max(x, y)}
tUnion = Primitive("Equal",arrow(ObjectSet, ObjectSet, ObjectSet), Union)

# [Do Some Counting]
def Count(x):return {"end":torch.sigmoid(x).sum(-1)}
tCount = Primitive("Count",arrow(ObjectSet, tint), Count)

def Equal(x):return lambda y:  {"end":8 * (.5 - (x - y).abs())}
tEqual = Primitive("Equal",arrow(treal, treal, Boolean), Equal)

# [Color category concepts]
color = Primitive("color", Concept, "color")
red = Primitive("red", Concept, "red")
green = Primitive("green",Concept, "green")
blue = Primitive("blue",Concept, "blue")

# [Shape category concepts]
shape = Primitive("shape", Attribute, "shape")
circle = Primitive("circle", Concept, "circle")
square = Primitive("square", Concept, "square")
diamond = Primitive("diamond", Concept, "diamond")

# [Relational Concepts]
left = Primitive("left", Concept, "left")
left = Primitive("right", Concept, "right")


p = Program.parse("(Exists $0)")
p = Program.parse("(Equal $1 $2)")
p = Program.parse("(Filter $0 red)")
p = Program.parse("(Relate $0 $0 left) ")

p = Program.parse("(Filter $0 red)")

config.concept_type = "cone"
executor = CentralExecutor(None, config)
end_scores = nn.Parameter(logit(torch.tensor([1.0,1.0,1.0,1.0,1.0])))
features = nn.Parameter(torch.randn([5,100]))

context = {"end":end_scores, "features":features, "executor":executor}
context["relations"] = executor.build_relations(context)


params = [{"params":executor.parameters()}]
optim = torch.optim.Adam(params, lr = 0.001)
history = []

epochs = 5000
itrs = epochs // 10
for epoch in range(epochs):
    loss = 0.0
    context["relations"] = executor.build_relations(context)

    p = Program.parse("(Filter $0 red)")
    out = p.evaluate({0:context})["end"]
    loss -= out[0]

    p = Program.parse("(Filter $0 square)")
    out = p.evaluate({0:context})["end"]
    loss -= out[2]

    p = Program.parse("(Relate $0 $0 right)")
    out = p.evaluate({0:context})["end"]
    loss -= out[1][2] - out[0].sum() - out[2:].sum() - out[1][:2].sum() - out[1][3:].sum()


    if epoch % itrs == 0:
        history.append(out.detach().unsqueeze(0).clone())
    optim.zero_grad()
    loss.backward()
    optim.step()
    sys.stdout.write("\repoch:{} loss:{}".format(epoch, loss))


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

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
        #plt.ylabel(concepts[attr])

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

p = Program.parse("(Equal (Count $0) (Count) )")
output = p.evaluate({0:context})["end"].unsqueeze(-1).detach().sigmoid()

full_analysis(context)