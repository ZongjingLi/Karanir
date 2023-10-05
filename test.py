from karanir.algs.graph import *
import matplotlib.pyplot as plt

from karanir.dklearn import *

from karanir.utils import save_im

save_im(torch.randn([64,64,3]).float().clamp(0.0,1.0).detach().numpy())

fcb = FCBlock(132,2,3,2)
fcb = FCBlock(132,2,3,2,activation = "nn.Sigmoid()")
fcb = FCBlock(132,2,3,2,
        activation = [
            "nn.Sigmoid()",
            "nn.Tanh()",
            "nn.Sigmoid()",
            "nn.Sigmoid()",
        ])
convnet = ConvolutionUnits(3, 32)

inputs = torch.randn([4,3,128,128])
outputs = convnet(inputs)
print(inputs.shape, outputs.shape)

inputs = torch.randn([10,3])
outputs = fcb(inputs)
print(inputs.shape, outputs.shape)

gridworld = GridGraph(16,32)

def test_metric(a,b):
    return abs(a["height"]-b["height"]) + 0.1

gridworld.metric = test_metric
for node in gridworld.nodes:
    gridworld.node_attributes[node]["height"] = np.random.randint(0,400)

dists, path = gridworld.bellman_ford((0,0), to=(12,22))

def gridworld_color_map(node):
    return node["height"]

render_results = gridworld.render(path,cmap = gridworld_color_map)

plt.imshow(render_results, cmap="rainbow")
plt.show()