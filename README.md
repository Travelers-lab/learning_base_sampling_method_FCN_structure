# Neural Network-Based Sampling for Path Planning

---

## Project Overview

**Learning-based sampling for path planning** leverages deep neural networks to predict optimal sampling distributions in grid-based environments. Unlike traditional uniform or heuristic sampling, this approach uses a fully convolutional network (FCN) to guide sampling towards regions likely to contain feasible or optimal paths, dramatically improving efficiency and success rates in algorithms like RRT and PRM.

**Key Features:**
- **Data-driven sampling:** Learns from thousands of synthetic maps and paths.
- **U-Net architecture:** Captures spatial context and local features for robust probability prediction.
- **Hybrid sampling:** Balances exploration (uniform) and exploitation (network-guided).
- **Modular pipeline:** From data generation to inference and visualization.
- **Extensible:** Easily adaptable to new environments, map sizes, or planning algorithms.

**Problem Statement:**  
Traditional random sampling in path planning can be inefficient, especially in cluttered or complex environments. This project proposes a neural network that, given a map, start, and goal, predicts a probability distribution for sampling, focusing computational effort where it matters most.

---

## Project Structure
learning_base_sampling_method_FCN_structure
├── README.md 
├── requirements.txt  
├── config/
│ ├── data_generation_config.yaml 
│ ├── model_config.yaml 
│ ├── training_config.yaml 
│ └── inference_config.yaml 
├── data/ 
│ ├── raw/ 
│ │ ├── maps/ 
│ │ ├── start_goal_pairs/ 
│ │ ├── paths/ 
│ │ ├── distance_maps/ 
│ │ └── heatmaps/ 
│ └── processed/ 
│ │ ├── train/ 
│ │ ├── val/ 
│ │ └── test/ 
├── src/ 
│ ├── data/ 
│ ├── models/ 
│ ├── training/ 
│ ├── inference/ 
│ └── utils/ 
├── scripts/ 
│ ├── create_project_structure.py 
│ ├── generate_dataset.py 
│ ├── dataset_building.py 
│ ├── train.py 
│ ├── sample_paths.py 
├── notebooks/ 
│ ├── data_exploration.ipynb 
│ ├── training_analysis.ipynb 
│ ├── sampling_visualization.ipynb 
│ └── comparison.ipynb 
└── results/ 
├── models/ 
│ ├── checkpoints/ 
│ └── best_model.pth 
├── logs/ │
 └── tensorboard/ 
 └── visualizations/ 
 ├── training_curves/ 
 ├── heatmap_predictions/ 
 ├── comparison_results/ 
 └── path_examples/

**Component Overview:**

| Directory/File         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `config/`             | YAML configuration files for all phases                                 |
| `data/raw/`           | Raw generated maps, paths, and labels                                   |
| `data/processed/`     | Ready-to-use datasets for training/validation/testing                   |
| `src/data/`           | Data generation and preprocessing modules                               |
| `src/models/`         | Model architecture and loss functions                                   |
| `src/training/`       | Training loop, optimizer, callbacks                                     |
| `src/inference/`      | Inference and sampling strategies                                       |
| `src/utils/`          | Visualization, metrics, and map processing utilities                    |
| `scripts/`            | Executable scripts for each pipeline stage                              |
| `notebooks/`          | Jupyter notebooks for analysis and visualization                        |
| `results/`            | Saved models, logs, and visual outputs                                  |

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Linux** (recommended; tested on Ubuntu)
- **System dependencies:**  
  - `numpy`, `scipy`, `matplotlib`, `opencv-python`, `torch`, `torchvision`, `pyyaml`, `tqdm`, `scikit-learn`, `tensorboard`, `jupyter`

### Installation

```bash
# Clone the repository
git clone git@github.com:Travelers-lab/learning_base_sampling_method_FCN_structure.git
cd learning_base_sampling_method_FCN_structure

# Install dependencies
pip install -r requirements.txt
```
## Quick Start
1. Generate Data
```bash
python generate_dataset.py
```
2. Split Dataset
```bash
python dataset_building.py
```
3. Train Model
```bash
python train.py
```
4. Run Inference & Sampling Demo
```bash
python sample_paths.py
```

## Usage Guide
### Data Generation
Configure parameters in 
```markdown 
config/data_generation_config.yaml.
```
Run generate_dataset.py to create synthetic maps, start/goal pairs, paths, distance transforms, and heatmaps.
Use dataset_building.py to split data into `train/val/test` sets.
### Model Training
Edit `config/model_config.yaml` and `config/training_config.yaml` for architecture and hyperparameters.
Run train.py to start training.
Checkpoints and logs are saved in `results/models/` and `results/logs/`.
### Inference & Sampling
Set parameters in `config/inference_config.yaml` (e.g., model path, lambda for hybrid sampling).
Use `sample_paths.py` to run inference and sample points for path planning.
Visualize results using scripts in `src/utils/visualization.py` or Jupyter notebooks.
### Configuration Files
 - `data_generation_config.yaml`: Map size, obstacle types, RRT parameters, dataset size.
 - 1model_config.yaml1: U-Net channels, batch norm, activation.
 - `raining_config.yaml`: Batch size, epochs, optimizer, scheduler, checkpointing.
 - `inference_config.yaml`: Model path, sampling lambda, device.
### Example:
``` bash
# inference_config.yaml
inference:
  model_path: "results/models/best_model.pth"
  device: "cpu"
sampling:
  lambda_prob: 0.7
```
## Results & Visualization
 - **Models:** Saved in `results/models/`
 - **Logs:** TensorBoard logs in `results/logs/tensorboard/`
 - **Visualizations:**
    * Training curves: `results/visualizations/training_curves/`
    * Predicted heatmaps:`results/visualizations/heatmap_predictions/`
    * Path examples: `results/visualizations/path_examples/`
 - **How to visualize:**
Use `src/utils/visualization.py` or Jupyter notebooks to plot probability maps, sampled points, and planned paths.
### Expected Output Example:
 - Probability heatmap showing high values along feasible paths.
 - Sampled points concentrated in promising regions.
 - Paths planned using hybrid sampling outperform uniform sampling.

## License
This project is licensed under the MIT License.