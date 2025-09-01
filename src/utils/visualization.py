import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(prob_map, title="Sampling Probability Map"):
    plt.figure(figsize=(6,6))
    plt.imshow(prob_map, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_sampling_points(obstacle_map, points, title="Sampled Points"):
    plt.figure(figsize=(6,6))
    plt.imshow(obstacle_map, cmap='gray')
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c='red', s=10)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_path(obstacle_map, path, start, goal, title="Planned Path"):
    plt.figure(figsize=(6,6))
    plt.imshow(obstacle_map, cmap='gray')
    if path is not None:
        px, py = zip(*path)
        plt.plot(px, py, 'b-', linewidth=2)
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
    plt.legend()
    plt.title(title)
    plt.axis('off')
    plt.show()