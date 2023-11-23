from karanir.thanagor import *

KFT = KaranirThanagor("The Domain")

from karanir.thanagor.types import *
from karanir.thanagor.program import *

def _plus1(x): return x + 1

def _addition(x): return lambda y: x + y


def _subtraction(x): return lambda y: x - y


def _division(x): return lambda y: x / y


subtraction = Primitive("-",
                        arrow(tint, arrow(tint, tint)),
                        _subtraction)
real_subtraction = Primitive("-.",
                             arrow(treal, treal, treal),
                             _subtraction)
addition = Primitive("+",
                     arrow(tint, arrow(tint, tint)),
                     Curried(_addition))
real_addition = Primitive("+.",
                          arrow(treal, treal, treal),
                          _addition)


def _multiplication(x): return lambda y: x * y


multiplication = Primitive("*",
                           arrow(tint, arrow(tint, tint)),
                           _multiplication)
real_multiplication = Primitive("*.",
                                arrow(treal, treal, treal),
                                _multiplication)
real_division = Primitive("/.",
                          arrow(treal, treal, treal),
                          _division)

plus_1 = Primitive("p1",arrow(tint, tint), _plus1)


def _power(a): return lambda b: a**b


real_power = Primitive("power",
                       arrow(treal, treal, treal),
                       _power)

k1 = Primitive("1", tint, 1)
k_negative1 = Primitive("negative_1", tint, -1)
k0 = Primitive("0", tint, 0)
for n in range(2,10):
    Primitive(str(n),tint,n)

f1 = Primitive("1.", treal, 1.)
f0 = Primitive("0.", treal, 0)
real = Primitive("REAL", treal, None)
fpi = Primitive("pi", treal, 3.14)

# Type Specification of ObjectSet, Attribute, Boolean and other apsects

ObjectSet = baseType("ObjectSet")
Attribute = baseType("Attribute")
Boolean = baseType("Boolean")
Concept = baseType("Concept")

def _Exists(objset):
    return torch.max(objset["end"] ** 2)

def _Filter(x):
    return lambda y: print(x["end"],y)


color = Primitive("color", Concept, "color")
red = Primitive("red", Concept, "red")
green = Primitive("green",Concept, "green")
blue = Primitive("blue",Concept, "blue")

shape = Primitive("shape", Attribute, "shape")
circle = Primitive("circle", Concept, "circle")
square = Primitive("square", Concept, "square")

tfilter = Primitive("Filter",arrow(ObjectSet, Attribute, ObjectSet), _Filter)
tExists = Primitive("Exists",arrow(ObjectSet, Boolean), _Exists)

p = Program.parse("(+ 1 $0)")
p = Program.parse("(Exists $0)")
#p = Program.parse("(Filter $0 red)")

#p = Program.parse("#(lambda (?? (+ 1 $0)) )")
context = 1
end_scores = nn.Parameter(torch.tensor([0.9,.7,.5,0.9,0.6]))
context = {"end":end_scores, "features":torch.randn([3,32]), "executor":None}
print(p, p.evaluate({0:context}))

params = [{"params":end_scores}]
optim = torch.optim.Adam(params, lr = 1e-1)
history = []
for epoch in range(32):
    loss = p.evaluate({0:context})

    history.append(end_scores.detach().unsqueeze(0).clone())
    optim.zero_grad()
    loss.backward()
    optim.step()

scores = torch.cat(history, dim = 0)


import matplotlib.pyplot as plt
fig = plt.figure("visualize track", figsize = (8,2))

ax = fig.add_subplot(111)
ax.imshow(scores.detach().permute(1,0).numpy())
plt.show()