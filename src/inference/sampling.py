import numpy as np

class HybridSampler:
    def __init__(self, lambda_prob=0.7):
        self.lambda_prob = lambda_prob

    def sample(self, prob_map, obstacle_map):
        free_space = np.where(obstacle_map == 0)
        if np.random.rand() < self.lambda_prob:
            # Importance sampling from prob_map
            flat_probs = prob_map.flatten()
            flat_probs[obstacle_map.flatten() == 1] = 0
            flat_probs /= flat_probs.sum()
            idx = np.random.choice(len(flat_probs), p=flat_probs)
            y, x = np.unravel_index(idx, prob_map.shape)
            return (x, y)
        else:
            # Uniform sampling from free space
            idx = np.random.randint(0, len(free_space[0]))
            return (free_space[1][idx], free_space[0][idx])