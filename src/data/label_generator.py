import numpy as np

class LabelGenerator:
    def __init__(self, size=100, sigma=5.0):
        self.size = size
        self.sigma = sigma

    def generate_heatmap(self, obstacle_map, path):
        heatmap = np.zeros((self.size, self.size))
        y, x = np.mgrid[0:self.size, 0:self.size]
        for px, py in path:
            dist2 = (x - px) ** 2 + (y - py) ** 2
            heatmap = np.maximum(heatmap, np.exp(-dist2 / (2 * self.sigma ** 2)))
        heatmap[obstacle_map == 1] = 0
        total = np.sum(heatmap)
        if total > 0:
            heatmap /= total
        return heatmap