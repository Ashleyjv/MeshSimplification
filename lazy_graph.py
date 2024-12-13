import numpy as np
import pyvista as pv
import networkx as nx
import random
from scipy.spatial import KDTree
import sys
import time
import heapq

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if False:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

class LazyGraph:
    def __init__(self, start, goal, mesh, 
                 step_size=1.0, 
                 max_iter=10000,
                 search_radius=2.0):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.mesh = mesh
        self.mpoints = mesh.points.tolist()
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.tree = nx.DiGraph()
        self.tree.add_node(tuple(self.start), parent=None)
        self.rtime = 0
        # For nearest neighbor search
        self.pq = []
        self.distances = {}
        self.visited = set()

        print("Mesh Bounds:", mesh.bounds)

        assert(self.is_collision_free(self.start, self.start))
        assert(self.is_collision_free(self.goal, self.goal))
    
    def is_collision_free(self, from_point, to_point):
        # Use PyVista's intersect_with_line method
        # It returns a tuple (points, ids)
        # If points is empty, there is no intersection
        # If points exist, check if the intersection is within the segment

        points = self.mesh.find_cells_intersecting_line(from_point, to_point)
        
        if points.size == 0:
            return True  # No collision
        # Calculate the distance from from_point to the first intersection point
        # If this distance is less than the distance between from_point and to_point,
        # there is a collision
        cell = self.mesh.get_cell(points[0])
        # print(cell)

        # Define segment start and end points
        y = np.array(from_point)
        z = np.array(to_point)
        direction = z - y

        if np.array_equal(y, z):
            bounds = cell.bounds
            ok = True
            for i in range(3):
                if bounds[2 * i] <= y[i] and y[i] <= bounds[2 * i + 1]:
                    pass
                else:
                    ok = False
            return not ok
        # direction = direction / np.linalg.norm(direction)

        tmin = 0.0
        tmax = 1.0

        for i in range(3):  # Iterate over x, y, z axes
            if direction[i] != 0:
                t1 = (cell.bounds[2*i] - y[i]) / direction[i]
                t2 = (cell.bounds[2*i+1] - y[i]) / direction[i]
                t_entry = min(t1, t2)
                t_exit = max(t1, t2)
                tmin = max(tmin, t_entry)
                tmax = min(tmax, t_exit)
                if tmin > tmax:
                    return True  # No collision within the segment
            else:
                if y[i] < cell.bounds[2*i] or y[i] > cell.bounds[2*i+1]:
                    return True  # Line parallel and outside the slab

        if tmin <= tmax and tmin <= 1 and tmax >= 0:
            # Intersection occurs within the segment
            return False  # Collision detected

        return True  # No collision within the segment

    def set_distance(self, node, dist):
        self.distances[tuple(node)] = dist

    def get_distance(self, node):
        return self.distances[tuple(node)]
    
    def dijkstra(self):
        """
        Builds the RRT tree using Dijkstra's algorithm.
        """
        debug("Dijkstra's algorithm started...")
        start_time = time.time()

        # Define the 6 cardinal directions
        dirs_3d = [
            (1,  0,  0),  # East
            (0,  1,  0),  # North
            (-1, 0,  0),  # West
            (0, -1,  0),  # South
            (0,  0,  1),  # Up
            (0,  0, -1)   # Down
        ]

        heapq.heappush(self.pq, (0, self.start))
        self.set_distance(self.start, 0)        

        while self.pq and self.max_iter > 0:
            current_dist, current_node = heapq.heappop(self.pq)

            if self.get_distance(current_node) != current_dist:
                continue

            # Check if goal is reached within search radius
            if np.linalg.norm(np.array(current_node) - np.array(self.goal)) <= self.search_radius:
                if self.is_collision_free(current_node, self.goal):
                    self.tree.add_node(self.goal, parent=current_node)
                    self.set_distance(self.goal, current_dist + np.linalg.norm(np.array(current_node) - np.array(self.goal)))
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.rtime = elapsed_time
                    debug(f"Runtime: {elapsed_time:.6f} seconds")
                    debug("Goal reached.")
                    return self.extract_path()

            # Explore all 6 cardinal directions
            for (dx, dy, dz) in dirs_3d:
                next_node = (
                    current_node[0] + dx * self.step_size,
                    current_node[1] + dy * self.step_size,
                    current_node[2] + dz * self.step_size
                )

                if next_node in self.visited:
                    continue

                if not self.is_collision_free(current_node, next_node):
                    continue  # Collision detected, skip this node

                new_dist = current_dist + self.distance(current_node, next_node)

                if next_node not in self.distances or new_dist < self.distances[next_node]:
                    self.set_distance(next_node, new_dist)
                    heapq.heappush(self.pq, (new_dist, next_node))
                    self.tree.add_node(next_node, parent=current_node)

            # Add direction towards the goal
            direction_to_goal = np.array(self.goal) - np.array(current_node)
            norm = np.linalg.norm(direction_to_goal)
            if norm != 0:
                direction_unit = direction_to_goal / norm
                next_node_goal = (
                    current_node[0] + direction_unit[0] * self.step_size,
                    current_node[1] + direction_unit[1] * self.step_size,
                    current_node[2] + direction_unit[2] * self.step_size
                )

                next_node_goal = tuple(next_node_goal)

                if next_node_goal not in self.visited:
                    if self.is_collision_free(current_node, next_node_goal):
                        new_dist_goal = current_dist + self.distance(current_node, next_node_goal)
                        if next_node_goal not in self.distances or new_dist_goal < self.distances[next_node_goal]:
                            self.distances[next_node_goal] = new_dist_goal
                            heapq.heappush(self.pq, (new_dist_goal, next_node_goal))
                            self.tree.add_node(next_node_goal, parent=current_node)

            self.max_iter -= 1

        debug("Reached maximum iterations without finding a path.")
        return None, None

    def distance(self, point1, point2):
        """
        Calculates Euclidean distance between two 3D points.
        """
        return np.linalg.norm(np.array(point2) - np.array(point1))

    def extract_path(self):
        path = []
        current = tuple(self.goal)
        dist = 0
        while self.tree.nodes[current]['parent'] is not None:
            parent = self.tree.nodes[current]['parent']
            path.append(current)
            path.append(parent)
            dist += np.linalg.norm(np.array(parent) - np.array(current))
            current = parent
        path.reverse()
        return path, dist
