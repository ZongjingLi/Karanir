from Karanir.algs.graph import *
import matplotlib.pyplot as plt

gridworld = GridGraph(10,20)

def test_metric(a,b):
    return abs(a["height"]-b["height"]) + 0.1

gridworld.metric = test_metric
for node in gridworld.nodes:
    gridworld.node_attributes[node]["height"] = np.random.randint(0,100)


dists, path = gridworld.bellman_ford((0,0), to=(6,18))


print(dists[(6,18)])

render_results = gridworld.render(path,color_map = None)

plt.imshow(render_results, cmap="winter")
plt.show()