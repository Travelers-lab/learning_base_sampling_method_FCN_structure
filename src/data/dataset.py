import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PathPlanningDataset(Dataset):
    def __init__(self, data_dir, augment=None):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        # Input: 4 channels
        x = np.stack([
            sample['obstacle_map'],
            sample['start_map'],
            sample['goal_map'],
            sample['distance_map']
        ], axis=0).astype(np.float32)
        y = sample['heatmap'].astype(np.float32)
        if self.augment:
            x, y = self.augment(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)