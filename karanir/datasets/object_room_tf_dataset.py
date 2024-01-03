import torch
from torch.utils.data import DataLoader, Dataset

import glob
import os


class ObjectRoomTFDataset(Dataset):
    def __init__(self, data_dir = None):
        super().__init__()
        self.file_list = glob.glob(os.path.join(data_dir,"object_room_tf",'*')) 

    def __len__(self):
        return 

    def __getitem__(self, idx):
        return 0