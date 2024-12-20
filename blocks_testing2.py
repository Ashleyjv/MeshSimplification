import pyvista as pv
import fast_simplification
import networkx as nx
import numpy as np
import random
import sys
from scipy.spatial import KDTree
from rrt_model import RRT
from lazy_graph import LazyGraph
import heapq
import json
import threading
import math

DEBUG = False

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if DEBUG:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

def visualize(mesh, simplified_mesh, orig_paths, simpl_paths):
    # Visualize
    debug("Visualizing paths...")
    plotter = pv.Plotter(shape=(1, 2))

    colors = ["red", "black", "orange", "purple"]
    # Original mesh with path
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color="blue", show_edges=True)
    # orig_paths = random.sample(orig_paths, min(len(orig_paths), 4))
    # for i in range(len(orig_paths)):
    #     if orig_paths[i] is None:
    #         continue
    lines = pv.line_segments_from_points(orig_paths)
    plotter.add_mesh(lines, color=colors[0])
    plotter.add_text("Original Mesh with Path", font_size=10)

    # Simplified mesh with path
    plotter.subplot(0, 1)
    plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
    # simpl_paths = random.sample(simpl_paths, min(len(simpl_paths), 4))
    # for i in range(len(simpl_paths)):
    #     if simpl_paths[i] is None:
    #         continue
    lines = pv.line_segments_from_points(simpl_paths)
    plotter.add_mesh(lines, color=colors[0])
    plotter.add_text("Simplified Mesh with Path", font_size=10)

    # Show the plot
    plotter.link_views()
    plotter.show()

def get_random_point(bounds):
    return np.array([
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ])

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

def get_graph_stats(points, faces, mesh, params):
    # Helper function to extract path points
    def get_path_points(path, pts):
        return np.array([pts[idx] for idx in path])

    max_height_constraint = 10.0
    # graph = create_graph(points, faces, max_height=max_height_constraint)
    stats = {
        "n_nodes" : len(points),
        "dijkstra" : [],
        "astar" : [],
        "rrt" : [],
        "rrtdist" : 0,
        "rrttime" : 0
    }

    path_points = -1
    
    # Validate path existence

    # Find shortest paths
    debug("Computing paths...")
    try:

        start = params["start"] #get_random_point(mesh.bounds) #
        goal = params["goal"]   #get_random_point(mesh.bounds) # 
        debug("Start:", start)
        debug("Goal:", goal)
        rrt = RRT(
            start=start,
            goal=goal,
            mesh=mesh,
            step_size=params["step_size"],
            max_iter=params["max_iter"],
            goal_sample_rate=0.18,
            search_radius=params["step_size"]
        )

        dijgraph = LazyGraph(
            start=start,
            goal=goal,
            mesh=mesh,
            step_size=params["step_size"],
            max_iter=params["max_iter"],
            search_radius=params["step_size"]
        )

        agraph = LazyGraph(
            start=start,
            goal=goal,
            mesh=mesh,
            step_size=params["step_size"],
            max_iter=params["max_iter"],
            search_radius=params["step_size"]
        )

        # Define target functions for threading
        def run_rrt():
            # if no rrt arg, skip
            path, dist = rrt.build_rrt()
            stats["rrttime"] = rrt.rtime
            stats["rrtdist"] = dist
            stats["rrt"] = path

        def run_astar():
            # if no a* arg, skip
            path, dist = agraph.a_star()
            stats["atime"] = agraph.rtime
            stats["adist"] = dist
            stats["astar"] = path

        def run_dijkstra():
            # if no dij arg, skip
            path, dist = dijgraph.dijkstra()
            stats["dijtime"] = dijgraph.rtime
            stats["dijdist"] = dist
            stats["dijkstra"] = path

        # Create threads
        thread_rrt = threading.Thread(target=run_rrt)
        thread_astar = threading.Thread(target=run_astar)
        thread_dijkstra = threading.Thread(target=run_dijkstra)

        # Start threads
        thread_rrt.start()
        thread_astar.start()
        thread_dijkstra.start()

        # Wait for both threads to finish
        thread_rrt.join()
        thread_astar.join()
        thread_dijkstra.join()

        print(stats)
        # need to measure runtime here
        # path = nx.dijkstra_path(graph, source=src_idx, target=dst_idx, weight=None)
        # path_points = get_path_points(path, points)
        # stats["dijkstra"].append(path_points)

        # path = nx.astar_path(graph, source=src_idx, target=dst_idx, weight=None)
        # path_points = get_path_points(path, points)
        # stats["astar"].append(path_points)
    except Exception as e:      # no path
        debug(e)
        stats["dijkstra"].append(None)
        stats["astar"].append(None)
    except AssertionError as e:
        debug(e)
        # return get_graph_stats(points, faces, mesh, queries)
    return stats


def get_mesh_stats(dataset_path, params):
    random.seed(params["seed"])

    original_mesh, original_points, nested_faces = load_mesh(dataset_path)

    simplified_mesh, simplified_points, simplified_faces, mapping = get_simplified_mesh(original_points, nested_faces)
    

    debug("Creating graphs for pathfinding...")

    original_stats = get_graph_stats(original_points, nested_faces, original_mesh, params)

    simplified_stats = get_graph_stats(simplified_points, simplified_faces, simplified_mesh, params)

    test_results = params | {
        "file" : dataset_path,
        "original" : original_stats,
        "simplified" : simplified_stats
    }

    with open(dataset_path + '_' + str(params['test_num']) + '.json', 'w') as file:
        json.dump(test_results, file, indent=4)
    print(test_results)
    visualize(original_mesh, simplified_mesh, original_stats["dijkstra"], simplified_stats["dijkstra"]) #simplified_stats["rrt"])

    # summarize stats using dictionaries

    # save stats into some csvs/graphs/images

# Path to the dataset
# dataset_path = "dataset/city.stl"

# queries = [(1897518, 1894639)]

# get_mesh_stats(dataset_path, queries)

test_suite = [
    (
        "dataset/city.stl", {
            "start" : [216, 678, 22],
            "goal" : [232, 646.6, 52.6],
            "max_iter" : 10000,
            "step_size" : 8,
            "seed" : 420,
            "test_num" : '1'
        }
    )
    # (
    #     "dataset/city.stl", {
    #         "start" : [345.61178723, 567.61959214,  76.7230741 ],
    #         "goal" : [270.5884959,  525.02272572,  74.67635382],
    #         "max_iter" : 10000,
    #         "step_size" : 8,
    #         "seed" : 420,
    #         "test_num" : '1'
    #     }
    # ),
    # (
    #     "dataset/building.stl", {
    #         "start" : [ 7.24923402, 17.4842314,   3.39377013],
    #         "goal" : [-9.49016435, -5.69409026,  1.61244157],
    #         "max_iter" : 5000,
    #         "step_size" : 1,
    #         "seed" : 23,
    #          "test_num" : 2
    #     }
    # ),
    # (
    #     "dataset/city.stl", {
    #         "start" : [284.07548918, 730.13227684,  55.99260382],
    #         "goal" : [298.8857376,  472.81611204,  10.10641103],
    #         "max_iter" : 5000,
    #         "step_size" : 3,
    #          "test_num" : 2
    #     }
    # ),
    # (
    #     "dataset/city.stl", {
    #         "start" : [294.2220246,  499.72158213,  62.09020357],
    #         "goal" : [270.5884959,  525.02272572,  74.67635382],
    #         "max_iter" : 5000,
    #         "step_size" : 3,
    #          "test_num" : 3
    #     }
    # ),
    # (
    #     "dataset/city.stl", {
    #         "start" : [345.61178723 567.61959214  76.7230741 ],
    #         "goal" : [270.5884959  525.02272572  74.67635382],
    #         "max_iter" : 5000,
    #         "step_size" : 3,
    #          "test_num" : 4
    #     }
    # )
    # (
    #     "dataset/some_other_file.stl", [
    #                             (..., ...)
    #                         ]
    # ),
]

# Can pass max_height arg for each dataset file
for (dataset_path, params) in test_suite:
    print("Running:", dataset_path, params)
    get_mesh_stats(dataset_path, params)


# 2 kinds of testing
# python3 testing.py
# this should execute all tests (itll take long time probs) and output 
# the summary table
# compare only relative distances/runtimes
# how much % improvement in runtime on avg between og/simplified for each algo
# how much % distance loss due to simplification? avg nums over all graphs
#               | original   | simplified
#  num_nodes    |            |
# dijkstra rtime|            |
# a* rtime      |            |
# rrt rtime     |            |
# dijk dist
# a* dist
# rrt dist

# python3 testing.py --viz i
# here i is test number indicated user wants to run visualization on ith test
# if so, run get_mesh_stats() and visualize() in it, add sm bool into get_mesh or somn
# shud output single test summarize table like this:
#               | original   | simplified
#  num_nodes    |            |
# dijkstra rtime|            |
# a* rtime      |            |
# rrt rtime     |            |
# dijk dist
# a* dist
# rrt dist