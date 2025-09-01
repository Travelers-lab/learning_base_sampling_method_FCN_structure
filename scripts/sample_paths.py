import os
import sys
from os.path import join, dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import numpy as np
import yaml
from src.inference.predictor import Predictor
from src.inference.sampling import HybridSampler
from src.utils.visualization import plot_heatmap, plot_sampling_points, plot_path

def main():
    # Load config
    with open("config/inference_config.yaml") as f:
        cfg = yaml.safe_load(f)
    lambda_prob = cfg['sampling']['lambda_prob']
    model_path = cfg['inference']['model_path']
    device = cfg['inference'].get('device', 'cpu')

    # Load a sample (replace with your own loading logic)
    sample = np.load("data/processed/test/sample_000023.npz")
    obstacle_map = sample['obstacle_map']
    start_map = sample['start_map']
    goal_map = sample['goal_map']
    distance_map = sample['distance_map']
    start = sample['start']
    goal = sample['goal']

    predictor = Predictor(model_path, device)
    prob_map = predictor.predict(obstacle_map, start_map, goal_map, distance_map)
    plot_heatmap(prob_map)

    sampler = HybridSampler(lambda_prob)
    points = [sampler.sample(prob_map, obstacle_map) for _ in range(100)]
    plot_sampling_points(obstacle_map, points)

    # Optionally, run path planner and plot path (not shown here)

if __name__ == "__main__":
    main()