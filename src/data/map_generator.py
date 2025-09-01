import numpy as np
import cv2
import random

class MapGenerator:
    def __init__(self, size=100, min_obstacle_size=10, max_obstacle_size=20, obstacle_types=None):
        self.size = size
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.obstacle_types = obstacle_types or ["triangle", "square", "circle"]

    def generate_random_shape(self, shape_type, center, size):
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if shape_type == "square":
            half = size // 2
            x_min = max(0, center[0] - half)
            x_max = min(self.size, center[0] + half)
            y_min = max(0, center[1] - half)
            y_max = min(self.size, center[1] + half)
            mask[y_min:y_max, x_min:x_max] = 1
        elif shape_type == "circle":
            cv2.circle(mask, center, size // 2, 1, -1)
        elif shape_type == "triangle":
            pts = []
            for _ in range(3):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(size // 4, size // 2)
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                x = np.clip(x, 0, self.size - 1)
                y = np.clip(y, 0, self.size - 1)
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def check_overlap(self, obstacle_map, new_obstacle):
        return np.any(obstacle_map & new_obstacle)

    def generate_map(self, num_obstacles=None):
        if num_obstacles is None:
            num_obstacles = np.random.randint(8, 16)
        obstacle_map = np.zeros((self.size, self.size), dtype=np.uint8)
        placed = 0
        attempts = 0
        while placed < num_obstacles and attempts < num_obstacles * 50:
            shape_type = random.choice(self.obstacle_types)
            size = np.random.randint(self.min_obstacle_size, self.max_obstacle_size + 1)
            margin = size // 2 + 2
            center = (np.random.randint(margin, self.size - margin),
                      np.random.randint(margin, self.size - margin))
            new_obstacle = self.generate_random_shape(shape_type, center, size)
            if not self.check_overlap(obstacle_map, new_obstacle):
                obstacle_map = np.logical_or(obstacle_map, new_obstacle).astype(np.uint8)
                placed += 1
            attempts += 1
        return obstacle_map

    def place_start_goal(self, obstacle_map, min_distance=50):
        free = np.where(obstacle_map == 0)
        free_indices = list(zip(free[1], free[0]))
        for _ in range(1000):
            s_idx, g_idx = np.random.choice(len(free_indices), 2, replace=False)
            start = free_indices[s_idx]
            goal = free_indices[g_idx]
            dist = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
            if dist >= min_distance:
                start_map = np.zeros_like(obstacle_map)
                goal_map = np.zeros_like(obstacle_map)
                start_map[start[1], start[0]] = 1
                goal_map[goal[1], goal[0]] = 1
                return start_map, goal_map, start, goal
        raise ValueError("Could not place start and goal with required distance.")