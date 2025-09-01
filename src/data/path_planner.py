import numpy as np
from dataclasses import dataclass
import random

@dataclass
class Node:
    x: float
    y: float
    parent: 'Node' = None

    def distance_to(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class RRTConnect:
    def __init__(self, obstacle_map, step_size=3.0, max_iterations=10000):
        self.map = obstacle_map
        self.size = obstacle_map.shape[0]
        self.step_size = step_size
        self.max_iterations = max_iterations

    def is_collision_free(self, n1, n2):
        x1, y1 = int(n1.x), int(n1.y)
        x2, y2 = int(n2.x), int(n2.y)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            if self.map[y, x]:
                return False
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def get_random_node(self):
        free = np.where(self.map == 0)
        idx = np.random.randint(0, len(free[0]))
        return Node(free[1][idx], free[0][idx])

    def get_nearest(self, tree, target):
        return min(tree, key=lambda n: n.distance_to(target))

    def steer(self, from_node, to_node):
        dist = from_node.distance_to(to_node)
        if dist <= self.step_size:
            return Node(to_node.x, to_node.y, from_node)
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(theta)
        new_y = from_node.y + self.step_size * np.sin(theta)
        return Node(new_x, new_y, from_node)

    def extend(self, tree, target):
        nearest = self.get_nearest(tree, target)
        new_node = self.steer(nearest, target)
        if self.is_collision_free(nearest, new_node):
            tree.append(new_node)
            if new_node.distance_to(target) < 1.0:
                return new_node, "Reached"
            else:
                return new_node, "Advanced"
        return nearest, "Trapped"

    def connect(self, tree, target):
        result = "Advanced"
        while result == "Advanced":
            _, result = self.extend(tree, target)
        return result

    def extract_path(self, tree1, tree2, node1, node2):
        path = []
        n = node1
        while n:
            path.append((int(n.x), int(n.y)))
            n = n.parent
        path = path[::-1]
        n = node2.parent
        while n:
            path.append((int(n.x), int(n.y)))
            n = n.parent
        return path

    def plan(self, start, goal):
        s_node = Node(start[0], start[1])
        g_node = Node(goal[0], goal[1])
        tree1 = [s_node]
        tree2 = [g_node]
        for _ in range(self.max_iterations):
            rand_node = self.get_random_node()
            new_node, status = self.extend(tree1, rand_node)
            if status != "Trapped":
                if self.connect(tree2, new_node) == "Reached":
                    return self.extract_path(tree1, tree2, new_node, tree2[-1])
            tree1, tree2 = tree2, tree1
        return None