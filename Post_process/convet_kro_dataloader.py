import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Kro_dataset(Dataset):

    def __init__(self, num_nodes):
        super(Kro_dataset, self).__init__()

        x1 = np.loadtxt('krodata/kroA%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
        x1 = x1 / (np.max(x1,0))
        x2 = np.loadtxt('krodata/kroB%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
        x2 = x2 / (np.max(x2,0))
        x = np.concatenate((x1, x2),axis=1)
        x = x.T
        x = x.reshape(1, 4, num_nodes)

        self.dataset = torch.from_numpy(x).float()
        self.dynamic = torch.zeros(1, 1, num_nodes)
        self.num_nodes = num_nodes
        self.size = 1


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])