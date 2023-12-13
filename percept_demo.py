from karanir.datasets.playroom_dataset import PlayroomDataset
from torch.utils.data import DataLoader

dataset_dir = "/Users/melkor/Documents/datasets/Playroom"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compute_flow",           default = False)
parser.add_argument("--precompute_flow",        default = False)
args = parser.parse_args(args = [])

dataset = PlayroomDataset(True, args, dataset_dir = dataset_dir, flow = False)
loader = DataLoader(dataset, batch_size = 2, shuffle = True)

for sample in loader:
    break;

import matplotlib.pyplot as plt

def visualize_sample(sample, fig_name = "visualize sample"):
    fig = plt.figure(fig_name)
    for i,k in enumerate(sample):
        ax = fig.add_subplot(1,3,i+1)
        ax.imshow(sample[k][0])
    plt.show()

visualize_sample(sample)