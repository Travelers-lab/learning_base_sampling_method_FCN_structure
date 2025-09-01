# path_sampling_nn/scripts/dataset_building.py

import os
import numpy as np
import shutil
import random
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_sample_ids(maps_dir):
    files = [f for f in os.listdir(maps_dir) if f.startswith("map_") and f.endswith(".npy")]
    ids = [int(f.split("_")[1].split(".")[0]) for f in files]
    ids.sort()
    return ids

def copy_sample(sample_id, raw_dir, processed_dir):
    npz_path = os.path.join(processed_dir, f"sample_{sample_id:06d}.npz")
    # Load all components
    map_file = os.path.join(raw_dir, "maps", f"map_{sample_id:06d}.npy")
    sg_file = os.path.join(raw_dir, "start_goal_pairs", f"sg_{sample_id:06d}.npy")
    path_file = os.path.join(raw_dir, "paths", f"path_{sample_id:06d}.npy")
    dist_file = os.path.join(raw_dir, "distance_maps", f"dist_{sample_id:06d}.npy")
    heat_file = os.path.join(raw_dir, "heatmaps", f"heat_{sample_id:06d}.npy")

    obstacle_map = np.load(map_file)
    sg_data = np.load(sg_file, allow_pickle=True).item()
    path = np.load(path_file, allow_pickle=True)
    distance_map = np.load(dist_file)
    heatmap = np.load(heat_file)

    np.savez_compressed(
        npz_path,
        obstacle_map=obstacle_map,
        start_map=sg_data["start_map"],
        goal_map=sg_data["goal_map"],
        start=sg_data["start"],
        goal=sg_data["goal"],
        path=path,
        distance_map=distance_map,
        heatmap=heatmap
    )

def main():
    config_path = os.path.join( "config", "data_generation_config.yaml")
    print(config_path)
    config = load_config(config_path)
    raw_dir = os.path.join( "data", "raw")
    processed_dir = os.path.join( "data", "processed")

    train_split = config["dataset_params"].get("train_split", 0.7)
    val_split = config["dataset_params"].get("val_split", 0.2)
    test_split = config["dataset_params"].get("test_split", 0.1)

    sample_ids = get_sample_ids(os.path.join(raw_dir, "maps"))
    random.shuffle(sample_ids)
    total = len(sample_ids)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    splits = {
        "train": sample_ids[:train_end],
        "val": sample_ids[train_end:val_end],
        "test": sample_ids[val_end:]
    }

    for split, ids in splits.items():
        split_dir = os.path.join(processed_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Processing {split} set: {len(ids)} samples")
        for sid in ids:
            copy_sample(sid, raw_dir, split_dir)

    print("Dataset splitting complete.")

if __name__ == "__main__":
    main()