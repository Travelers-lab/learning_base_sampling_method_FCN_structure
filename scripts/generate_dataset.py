import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.map_generator import MapGenerator
from src.data.path_planner import RRTConnect
from src.data.label_generator import LabelGenerator
from src.utils.map_processing import compute_distance_transform

def main():
    with open("config/data_generation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    size = cfg["map_params"]["size"]
    num_obstacles_range = cfg["map_params"]["num_obstacles_range"]
    obstacle_size_range = cfg["map_params"]["obstacle_size_range"]
    obstacle_types = cfg["map_params"]["obstacle_types"]
    min_distance = cfg["start_goal_params"]["min_distance"]
    step_size = cfg["path_planning_params"]["step_size"]
    max_iterations = cfg["path_planning_params"]["max_iterations"]
    sigma = cfg["label_params"]["gaussian_sigma"]
    num_samples = cfg["dataset_params"]["num_samples"]

    map_gen = MapGenerator(size, obstacle_size_range[0], obstacle_size_range[1], obstacle_types)
    label_gen = LabelGenerator(size, sigma)

    for idx in tqdm(range(num_samples)):
        try:
            num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
            obstacle_map = map_gen.generate_map(num_obstacles)
            start_map, goal_map, start, goal = map_gen.place_start_goal(obstacle_map, min_distance)
            planner = RRTConnect(obstacle_map, step_size, max_iterations)
            path = planner.plan(start, goal)
            if path is None:
                continue
            heatmap = label_gen.generate_heatmap(obstacle_map, path)
            dist_map = compute_distance_transform(obstacle_map)

            np.save(f"data/raw/maps/map_{idx:06d}.npy", obstacle_map)
            np.save(f"data/raw/start_goal_pairs/sg_{idx:06d}.npy", {"start_map": start_map, "goal_map": goal_map, "start": start, "goal": goal})
            np.save(f"data/raw/paths/path_{idx:06d}.npy", np.array(path))
            np.save(f"data/raw/distance_maps/dist_{idx:06d}.npy", dist_map)
            np.save(f"data/raw/heatmaps/heat_{idx:06d}.npy", heatmap)
        except Exception as e:
            print(f"Sample {idx}: Error - {e}")

if __name__ == "__main__":
    main()