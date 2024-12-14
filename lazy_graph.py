import numpy as np
import pyvista as pv
import networkx as nx
import random
from scipy.spatial import KDTree
import sys
import time
import heapq
import math

EPS = 0.8

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if False:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

def create_cell_bins(mesh, step_sz):
    # Extract mesh bounds
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    print(f"Mesh x-range: [{xmin}, {xmax}]")

    # Calculate the number of bins needed
    num_bins = math.ceil((xmax - xmin) / step_sz)
    print(f"Number of bins: {num_bins}")

    # Initialize a list to hold cell indices for each bin
    cell_bins_indices = [[] for _ in range(num_bins)]

    cell_centroids = mesh.cell_centers().points

    for cell_id in range(mesh.n_cells):
        centroid_x = cell_centroids[cell_id][0]  # Extract x-coordinate of centroid
        bin_idx = int((centroid_x - xmin) / step_sz)
        if bin_idx == num_bins:
            bin_idx -= 1
        bin_idx = max(0, min(bin_idx, num_bins - 1))
        cell_bins_indices[bin_idx].append(cell_id)

    cell_bins = []
    for i in range(num_bins):
        cell_indices = cell_bins_indices[i]
        if not cell_indices:
            cell_bins.append(pv.PolyData())
            continue
        extracted = mesh.extract_cells(cell_indices)
        extracted.clean()
        cell_bins.append(extracted)
        debug(f"Bin {i}: x-range=[{xmin + i * step_sz}, {xmin + (i + 1) * step_sz}) - Cells: {len(cell_indices)}")

    return cell_bins

def get_relevant_bins(x, y, xmin, step_sz, num_bins):
    x_start = min(x[0], y[0])
    x_end = max(x[0], y[0])

    # Calculate bin indices
    bin_start = int((x_start - xmin) / step_sz)
    bin_end = int((x_end - xmin) / step_sz)

    # Handle edge cases
    bin_start = max(0, min(bin_start, num_bins - 1))
    bin_end = max(0, min(bin_end, num_bins - 1))

    # Collect all bin indices that overlap with the line segment's x-range
    relevant_bin_indices = range(bin_start, bin_end + 1)

    return relevant_bin_indices

class LazyGraph:
    def __init__(self, start, goal, mesh, 
                 step_size=1.0, 
                 max_iter=10000,
                 search_radius=2.0):

        self.start = tuple(start)
        self.goal = tuple(goal)
        self.bounds = mesh.bounds
        self.mesh_bins = create_cell_bins(mesh, step_size)
        self.mpoints = mesh.points.tolist()
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.tree = nx.DiGraph()
        self.tree.add_node(tuple(self.start), parent=None)
        self.rtime = 0
        # For nearest neighbor search
        self.kdtree = KDTree([self.start])
        self.pq = []
        self.distances = {}

        print("Mesh Bounds:", self.bounds)

        assert(self.is_collision_free(self.start, self.start))
        assert(self.is_collision_free(self.goal, self.goal))
    
    def upd_tree(self):
        self.kdtree = KDTree(list(self.distances.keys()))

    def collision_free(self, from_point, to_point, index):
        # Use PyVista's intersect_with_line method
        # It returns a tuple (points, ids)
        # If points is empty, there is no intersection
        # If points exist, check if the intersection is within the segment

        points = self.mesh_bins[index].find_cells_intersecting_line(from_point, to_point)
        
        if points.size == 0:
            return True  # No collision
        # Calculate the distance from from_point to the first intersection point
        # If this distance is less than the distance between from_point and to_point,
        # there is a collision
        cell = self.mesh_bins[index].get_cell(points[0])
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

    def is_collision_free(self, from_point, to_point):
        bins = get_relevant_bins(from_point, to_point, self.bounds[0], 
                                self.step_size, len(self.mesh_bins))
        for i in bins:
            if not self.collision_free(from_point, to_point, i):
                return False
        return True

    def set_distance(self, node, dist):
        self.distances[tuple(node)] = dist

    def get_distance(self, node):
        return self.distances[tuple(node)]
    
    def fix_point(self, point):
        new_pt = []
        for i in range(3):
            new_pt.append(round(point[i], 1))
        point = tuple(new_pt)
        nearest_pt = self.kdtree.query(point)[0]
        if self.distance(point, nearest_pt) <= EPS:
            return nearest_pt
        return point

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
        itr = 0

        while self.pq and itr < self.max_iter:

            if itr < 100 or itr % 200 == 0:
                self.upd_tree()

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

                next_node = self.fix_point(next_node)

                if not self.is_collision_free(current_node, next_node):
                    continue  # Collision detected, skip this node

                new_dist = current_dist + self.distance(current_node, next_node)

                if next_node not in self.distances or new_dist < self.get_distance(next_node):
                    self.set_distance(next_node, new_dist)
                    debug(next_node, " added.")
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

                next_node_goal = self.fix_point(next_node_goal)

                if self.is_collision_free(current_node, next_node_goal):
                    new_dist_goal = current_dist + self.distance(current_node, next_node_goal)
                    if next_node_goal not in self.distances or new_dist_goal < self.distances[next_node_goal]:
                        self.set_distance(next_node_goal, new_dist_goal)
                        debug(next_node_goal, " added.")
                        heapq.heappush(self.pq, (new_dist_goal, next_node_goal))
                        self.tree.add_node(next_node_goal, parent=current_node)

            itr += 1

        debug("Reached maximum iterations without finding a path.")
        return None, None

    def a_star(self):
        debug("A* algorithm started...")
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

        heapq.heappush(self.pq, (0 + self.heuristic(self.start), self.start))
        self.set_distance(self.start, 0)

        itr = 0
        while self.pq and itr < self.max_iter:
            current_f, current_node = heapq.heappop(self.pq)

            dist = current_f - self.heuristic(current_node)

            if self.get_distance(current_node) != dist:
                continue

            if itr < 100 or itr % 200 == 0:
                self.upd_tree()

            # Check if goal is reached within search radius
            distance_to_goal = self.distance(current_node, self.goal)
            if distance_to_goal <= self.search_radius:
                if self.is_collision_free(current_node, self.goal):
                    self.tree.add_node(self.goal, parent=current_node)
                    final_dist = dist + distance_to_goal
                    self.set_distance(self.goal, final_dist)
                    debug("Goal reached.")
                    self.rtime = time.time() - start_time
                    path, total_distance = self.extract_path()
                    return path, total_distance

            # Explore all 6 cardinal directions
            for (dx, dy, dz) in dirs_3d:
                next_node = (
                    current_node[0] + dx * self.step_size,
                    current_node[1] + dy * self.step_size,
                    current_node[2] + dz * self.step_size
                )

                next_node = self.fix_point(next_node)

                if not self.is_collision_free(current_node, next_node):
                    continue  # Collision detected, skip this node

                tentative_g = dist + self.distance(current_node, next_node)

                if next_node not in self.distances or tentative_g < self.get_distance(next_node):
                    self.set_distance(next_node, tentative_g)
                    f_score = tentative_g + self.heuristic(next_node)
                    heapq.heappush(self.pq, (f_score, next_node))
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

                next_node_goal = self.fix_point(next_node_goal)

                if self.is_collision_free(current_node, next_node_goal):
                    tentative_g_goal = dist + self.distance(current_node, next_node_goal)
                    if next_node_goal not in self.distances or tentative_g_goal < self.get_distance(next_node_goal):
                        self.set_distance(next_node_goal, tentative_g_goal)
                        f_score_goal = tentative_g_goal + self.heuristic(next_node_goal)
                        heapq.heappush(self.pq, (f_score_goal, next_node_goal))
                        self.tree.add_node(next_node_goal, parent=current_node)

            itr += 1

        debug("Reached maximum iterations without finding a path.")
        self.rtime = time.time() - start_time
        return None, None

    def distance(self, point1, point2):
        """
        Calculates Euclidean distance between two 3D points.
        """
        return np.linalg.norm(np.array(point2) - np.array(point1))

    def heuristic(self, node):
        return np.linalg.norm(np.array(node) - np.array(self.goal))

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

