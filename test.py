from Karanir.algs.graph import *
import matplotlib.pyplot as plt

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