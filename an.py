from karanir.thanagor import *
from karanir.thanagor.knowledge.executor import CentralExecutor

KFT = KaranirThanagor("The Domain", config)
KFT.print_summary()


import sys

from karanir.thanagor.dsl.vqa_primitives import *

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
optim = torch.optim.Adam(params, lr = 0.01)
history = []

epochs = 3000
itrs = epochs // 10
for epoch in range(epochs):
    loss = 0.0
    context["relations"] = executor.build_relations(context)

    p = Program.parse("(Filter $0 red)")
    out = p.evaluate({0:context})["end"]
    loss -= out[0] - out[1:-1].sum()

    p = Program.parse("(Filter $0 square)")
    out = p.evaluate({0:context})["end"]
    loss -= out[2] - out[3:].sum() - out[:2].sum()

    p = Program.parse("(Relate $0 $0 left)")
    out = p.evaluate({0:context})["end"]
    loss -= out[1][2] - out[0].sum() - out[2:].sum() - out[1][:2].sum() - out[1][3:].sum()


    if epoch % itrs == 0:
        history.append(out.detach().unsqueeze(0).clone())
    optim.zero_grad()
    loss.backward()
    optim.step()
    sys.stdout.write("\repoch:{} loss:{}".format(epoch, loss))



p = Program.parse("(Count (Filter $0 red))")
print(p)

output = p.evaluate({0:context})["end"].unsqueeze(-1).detach()
print(output)

p = Program.parse("(Count (Filter $0 square))")
print(p)

output = p.evaluate({0:context})["end"].unsqueeze(-1).detach()
print(output)


from karanir.thanagor.utils import full_analysis
full_analysis(context)
