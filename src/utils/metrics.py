import numpy as np

def kl_divergence(p, q):
    p = p.flatten()
    q = q.flatten()
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def compare_sampling_efficiency(paths_nn, paths_uniform):
    nn_lengths = [len(p) for p in paths_nn if p is not None]
    uniform_lengths = [len(p) for p in paths_uniform if p is not None]
    return np.mean(nn_lengths), np.mean(uniform_lengths)