import os
import sys
from os.path import join, dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
print(os.path)
import yaml
import torch
from src.models.unet import UNet
from src.models.loss import CrossEntropy2d
from src.data.dataset import PathPlanningDataset
from src.data.augmentation import random_flip_rotate
from src.training.trainer import train_model

def main():
    with open("config/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)['model']
    with open("config/training_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**model_cfg).to(device)
    loss_fn = CrossEntropy2d()

    train_set = PathPlanningDataset("data/processed/train", augment=random_flip_rotate)
    val_set = PathPlanningDataset("data/processed/val")
    save_dir = "results"

    train_model(model, train_set, val_set, loss_fn, train_cfg, device, save_dir)

if __name__ == "__main__":
    main()