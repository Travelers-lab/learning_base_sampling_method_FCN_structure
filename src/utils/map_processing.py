import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_distance_transform(obstacle_map):
    return distance_transform_edt(obstacle_map == 0)