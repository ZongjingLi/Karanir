from Karanir.algs.graph import *
import matplotlib.pyplot as plt

gridworld = GridGraph(10,20)

def test_metric(a,b):
    return abs(a["height"]-b["height"]) + 0.1

gridworld.metric = test_metric
for node in gridworld.nodes:
    gridworld.node_attributes[node]["height"] = np.random.randint(0,100)

dists, path = gridworld.bellman_ford((0,0), to=(6,18))

def gridworld_color_map(node):
    return node["height"]

render_results = gridworld.render(path,cmap = gridworld_color_map)

plt.imshow(render_results, cmap="winter")
plt.show()