from karanir.thanagor import *

KFT = KaranirThanagor("The Domain")

print(KFT)

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

ObjectSet = baseType("ObjectSet")
Attribute = baseType("Attribute")
Boolean = baseType("Boolean")

def _Exists(objset):
    return torch.max(objset["end"])

def _Filter(x):
    return lambda y: print(x["end"],y)


red = Primitive("red", Attribute, "red_color")

tfilter = Primitive("Filter",arrow(ObjectSet, Attribute, ObjectSet), _Filter)
tExists = Primitive("Exists",arrow(ObjectSet, Boolean), _Exists)

p = Program.parse("(+ 1 $0)")
p = Program.parse("(Exists $0)")
p = Program.parse("(Filter $0 red)")
#p = Program.parse("#(lambda (?? (+ 1 $0)) )")
context = 1
context = {"end":torch.tensor([0.9,.7,.5]), "features":torch.randn([3,32]), "executor":None}
print(p, p.evaluate({0:context}))
