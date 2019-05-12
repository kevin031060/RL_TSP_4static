from tasks.motsp import TSPDataset, reward
from torch.utils.data import DataLoader
import torch

train_data = TSPDataset(10, 10000, 1234)
train_loader = DataLoader(train_data, 100, True, num_workers=0)
iter_data = iter(train_loader)
batch = iter_data.next()[0]
print(reward(batch, torch.randperm(10).expand(1,10), 1, 0))

