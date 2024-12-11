import pyvista as pv
import fast_simplification
import networkx as nx
import numpy as np
import random
import sys
from scipy.spatial import KDTree
from rrt_model import RRT

DEBUG = True

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if DEBUG:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

def load_mesh(dataset_path):
        # Load the STL file
    debug("Loading city STL dataset...")
    original_mesh = pv.read(dataset_path)

    # Extract points and faces for graph creation and simplification
    original_points = original_mesh.points.tolist()
    original_faces = original_mesh.faces.reshape(-1, 4)[:, 1:]  # Extract triangle indices
    nested_faces = original_faces.tolist()
    (xmin, xmax, ymin, ymax, zmin, zmax) = original_mesh.bounds


    return original_mesh, original_points, nested_faces

def get_simplified_mesh(original_points, nested_faces, tg_red=0.5):
    # Simplify the mesh
    debug("Simplifying the city mesh...")
    simplified_points, simplified_faces, mapping = fast_simplification.simplify(
        original_points,
        nested_faces,
        target_reduction=tg_red,  # Adjust as needed
        return_collapses=True
    )

    debug("Num of pts in original:", len(original_points))
    debug("Num of pts in simplified:", len(simplified_points))

    # Reformat simplified faces for PyVista
    flat_simplified_faces = []
    for face in simplified_faces:
        flat_simplified_faces.append(len(face))  # Number of vertices in the face
        flat_simplified_faces.extend(face)

    # Create simplified PyVista mesh
    simplified_mesh = pv.PolyData(simplified_points, flat_simplified_faces)

    return simplified_mesh, simplified_points, simplified_faces, mapping

# Function to create a graph from a mesh
def create_graph(points, faces, max_height=None):
    graph = nx.Graph()
    for i, point in enumerate(points):
        graph.add_node(i, pos=tuple(point))
    for face in faces:
        # print(len(face), face)
        for j in range(len(face)):
            # Connect vertices if the height constraint is met
            if max_height is None or abs(points[face[j]][2] - points[face[(j + 1) % len(face)]][2]) <= max_height:
                graph.add_edge(face[j], face[(j + 1) % len(face)])  # Connect vertices cyclically
    return graph

def get_graph_stats(points, faces, mesh, queries):
    # Helper function to extract path points
    def get_path_points(path, pts):
        return np.array([pts[idx] for idx in path])

    max_height_constraint = 10.0
    # graph = create_graph(points, faces, max_height=max_height_constraint)
    stats = {
        "n_nodes" : 0,
        "dijkstra" : [],
        "astar" : [],
        "rrt" : []
    }
    for (src_idx, dst_idx) in queries:
        path_points = -1
        
        # Validate path existence

        # Find shortest paths

        src_idx = random.randint(0, len(points) - 1)
        dst_idx = random.randint(0, len(points) - 1)
        debug("Computing paths...")
        try:

            debug("Start:", src_idx, points[src_idx])
            debug("Goal:", src_idx, points[src_idx])
            rrt = RRT(
                start=points[src_idx],
                goal=points[dst_idx],
                mesh=mesh,
                step_size=5.0,
                max_iter=5000,
                goal_sample_rate=0.1,
                search_radius=5.0
            )

            # Build RRT and extract path
            path = rrt.build_rrt()
            print(path)
            stats["rrt"].append(path)
            # need to measure runtime here
            # path = nx.dijkstra_path(graph, source=src_idx, target=dst_idx, weight=None)
            # path_points = get_path_points(path, points)
            # stats["dijkstra"].append(path_points)

            # path = nx.astar_path(graph, source=src_idx, target=dst_idx, weight=None)
            # path_points = get_path_points(path, points)
            # stats["astar"].append(path_points)
        except nx.NetworkXNoPath as e:      # no path
            debug(e)
            stats["dijkstra"].append(None)
            stats["astar"].append(None)
        except AssertionError as e:
            debug(e)
            return get_graph_stats(points, faces, mesh, queries)
        # path = nx.dijkstra_path(graph, source=src_idx, target=dst_idx, weight=None)

        # path_points = get_path_points(path, points)

        # stats["dijkstra"].append(path_points)
    return stats

def visualize(mesh, simplified_mesh, orig_paths, simpl_paths):
    # Visualize
    debug("Visualizing paths...")
    plotter = pv.Plotter(shape=(1, 2))

    colors = ["red", "black", "orange", "purple"]
    # Original mesh with path
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color="blue", show_edges=True)
    orig_paths = random.sample(orig_paths, min(len(orig_paths), 4))
    for i in range(len(orig_paths)):
        if orig_paths[i] is None:
            continue
        plotter.add_lines(orig_paths[i], color=colors[i], width=51)
    plotter.add_text("Original Mesh with Path", font_size=10)

    # Simplified mesh with path
    plotter.subplot(0, 1)
    plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
    simpl_paths = random.sample(simpl_paths, min(len(simpl_paths), 4))
    for i in range(len(simpl_paths)):
        if simpl_paths[i] is None:
            continue
        plotter.add_lines(simpl_paths[i], color=colors[i], width=15)
    plotter.add_text("Simplified Mesh with Path", font_size=10)

    # Show the plot
    plotter.link_views()
    plotter.show()

def get_mapped_queries(simpl_pts, og_pts, mapping, queries):
    # Create a KDTree for nearest neighbor mapping
    simplified_tree = KDTree(simpl_pts)

    # Helper function to remap isolated nodes
    def get_mapped_or_nearest(node, mapping, simplified_points):
        if node < len(mapping) and mapping[node] != -1:  # Check if node is mapped
            return mapping[node]
        # Fallback to nearest node in simplified points
        original_point = og_pts[node]
        nearest_idx = simplified_tree.query(original_point)[1]
        print(f"Node {node} not mapped, using nearest node {nearest_idx}")
        return nearest_idx

    # Select source and destination nodes for pathfinding
    mapped_queries = []
    for (src_idx, dst_idx) in queries:
        simplified_src_idx = get_mapped_or_nearest(src_idx, mapping, simpl_pts)
        simplified_dst_idx = get_mapped_or_nearest(dst_idx, mapping, simpl_pts)
        mapped_queries.append((simplified_src_idx, simplified_dst_idx))
    return mapped_queries

def get_mesh_stats(dataset_path, queries):
    original_mesh, original_points, nested_faces = load_mesh(dataset_path)

    simplified_mesh, simplified_points, simplified_faces, mapping = get_simplified_mesh(original_points, nested_faces)
    
    debug("Creating graphs for pathfinding...")

    original_stats = get_graph_stats(original_points, nested_faces, original_mesh, queries)

    mapped_queries = get_mapped_queries(simplified_points, original_points, mapping, queries)

    simplified_stats = get_graph_stats(simplified_points, simplified_faces, simplified_mesh, mapped_queries)

    if DEBUG:
        visualize(original_mesh, simplified_mesh, original_stats["rrt"], simplified_stats["rrt"])

    # summarize stats using dictionaries

    # save stats into some csvs/graphs/images

# Path to the dataset
# dataset_path = "dataset/city.stl"

# queries = [(1897518, 1894639)]

# get_mesh_stats(dataset_path, queries)

test_suite = [
    (
        "dataset/city.stl", [
                                (1897518, 1894639)
                            ]
    ),
    # (
    #     "dataset/some_other_file.stl", [
    #                             (..., ...)
    #                         ]
    # ),
]

# Can pass max_height arg for each dataset file
for (dataset_path, queries) in test_suite:
    get_mesh_stats(dataset_path, queries)