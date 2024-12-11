import numpy as np
import pyvista as pv
import networkx as nx
import random
from scipy.spatial import KDTree
import sys

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if True:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

class RRT:
    def __init__(self, start, goal, mesh, 
                 step_size=1.0, 
                 max_iter=100000, 
                 goal_sample_rate=0.05, 
                 search_radius=2.0):
        """
        Initializes the RRT planner.
        
        :param start: Tuple or list with start coordinates (x, y, z).
        :param goal: Tuple or list with goal coordinates (x, y, z).
        :param mesh: PyVista mesh object representing obstacles.
        :param step_size: Maximum distance between nodes.
        :param max_iter: Maximum number of iterations to run RRT.
        :param goal_sample_rate: Probability of sampling the goal.
        :param search_radius: Radius to consider goal reached.
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.mesh = mesh
        self.mpoints = mesh.points.tolist()
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.tree = nx.DiGraph()
        self.tree.add_node(tuple(self.start), parent=None)
        
        # For nearest neighbor search
        self.kdtree = KDTree([self.start])
        self.nodes = [self.start]

        print("Mesh Bounds:", mesh.bounds)

        assert(self.is_collision_free(self.start, self.start))
        assert(self.is_collision_free(self.goal, self.goal))
    
    def get_random_point(self, bounds, rate):
        if random.random() < rate:
            return self.goal
        else:
            return np.array([
                random.uniform(bounds[0], bounds[1]),
                random.uniform(bounds[2], bounds[3]),
                random.uniform(bounds[4], bounds[5])
            ])
    
    def nearest_neighbor(self, point):
        distance, index = self.kdtree.query(point)
        return index
    
    def steer(self, from_point, to_point):
        """
        Steers from from_point towards to_point by step_size.
        """
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance == 0:
            return from_point
        direction = direction / distance
        distance = min(self.step_size, distance)
        new_point = from_point + distance * direction
        return new_point
    
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
        print(self.mesh.get_cell(points[0]))
        

        # Define segment start and end points
        y = np.array(from_point)
        z = np.array(to_point)
        direction = z - y

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


        print(self.mesh.get_cell(points[0]), random_pt)
        intersection_distance = np.linalg.norm(random_pt - from_point)
        total_distance = np.linalg.norm(to_point - from_point)
        
        if intersection_distance < total_distance:
            return False  # Collision detected
        return True  # No collision
    
    def build_rrt(self):
        """
        Builds the RRT tree.
        """
        debug("RRT started...")
        bounds = self.mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        for i in range(self.max_iter):
            rand_point = self.get_random_point(bounds, self.goal_sample_rate)
            nearest_idx = self.nearest_neighbor(rand_point)
            nearest_point = self.nodes[nearest_idx]
            new_point = self.steer(nearest_point, rand_point)
            
            if self.is_collision_free(nearest_point, new_point):
                self.tree.add_node(tuple(new_point), parent=tuple(nearest_point))
                self.nodes.append(new_point)
                self.kdtree = KDTree(self.nodes)  # Update KDTree with the new node

                debug(new_point, " added. KD Tree updated.")
                # Check if goal is reached
                if np.linalg.norm(new_point - self.goal) <= self.search_radius:
                    if self.is_collision_free(new_point, self.goal):
                        self.tree.add_node(tuple(self.goal), parent=tuple(new_point))
                        debug(f"Goal reached in {i+1} iterations.")
                        return self.extract_path()
        
        debug("Reached maximum iterations without finding a path.")
        return None
    
    def extract_path(self):
        path = [tuple(self.goal)]
        current = tuple(self.goal)
        while self.tree.nodes[current]['parent'] is not None:
            parent = self.tree.nodes[current]['parent']
            path.append(parent)
            current = parent
        path.reverse()
        return path
