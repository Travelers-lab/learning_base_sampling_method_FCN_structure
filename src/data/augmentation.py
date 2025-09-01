import numpy as np
import random
import torch

def random_flip_rotate(x, y):
    # x: (C, H, W), y: (H, W)
    k = random.randint(0, 3)
    x = np.rot90(x, k, axes=(1,2))
    y = np.rot90(y, k)
    if random.random() > 0.5:
        x = np.flip(x, axis=2)
        y = np.flip(y, axis=1)
    if random.random() > 0.5:
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=0)
    return x.copy(), y.copy()