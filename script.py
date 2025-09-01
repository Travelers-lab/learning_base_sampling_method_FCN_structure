import os

DIRS = [
    "config",
    "data/raw/maps",
    "data/raw/start_goal_pairs",
    "data/raw/paths",
    "data/raw/distance_maps",
    "data/raw/heatmaps",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "src/data",
    "src/models",
    "src/training",
    "src/inference",
    "src/utils",
    "scripts",
    "notebooks",
    "results/models/checkpoints",
    "results/logs/tensorboard",
    "results/visualizations/training_curves",
    "results/visualizations/heatmap_predictions",
    "results/visualizations/comparison_results",
    "results/visualizations/path_examples",
]

INIT_FILES = [
    "__init__.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/models/__init__.py",
    "src/training/__init__.py",
    "src/inference/__init__.py",
    "src/utils/__init__.py",
]

GITKEEP = [
    "data/raw/maps/.gitkeep",
    "data/raw/start_goal_pairs/.gitkeep",
    "data/raw/paths/.gitkeep",
    "data/raw/distance_maps/.gitkeep",
    "data/raw/heatmaps/.gitkeep",
    "data/processed/train/.gitkeep",
    "data/processed/val/.gitkeep",
    "data/processed/test/.gitkeep",
    "results/models/checkpoints/.gitkeep",
    "results/logs/tensorboard/.gitkeep",
    "results/visualizations/training_curves/.gitkeep",
    "results/visualizations/heatmap_predictions/.gitkeep",
    "results/visualizations/comparison_results/.gitkeep",
    "results/visualizations/path_examples/.gitkeep",
]

def main():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
    for f in INIT_FILES:
        with open(f, "w") as fp:
            fp.write("# Package init\n")
    for f in GITKEEP:
        with open(f, "w") as fp:
            fp.write("")
    print("Project structure created.")

if __name__ == "__main__":
    main()