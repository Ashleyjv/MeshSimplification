import argparse
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
import os

DEBUG = True

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    if DEBUG:
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

def get_random_point(bounds):
    return np.array([
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ])

def save_mesh_data(file_path, data):
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_data = convert_to_serializable(data)
    with open(file_path, 'w') as file:
        json.dump(serializable_data, file, indent=4)

def load_mesh_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return None

def load_mesh(dataset_path):
    debug("Loading city STL dataset...")
    original_mesh = pv.read(dataset_path)
    original_points = original_mesh.points.tolist()
    original_faces = original_mesh.faces.reshape(-1, 4)[:, 1:].tolist()
    return original_mesh, original_points, original_faces

def get_simplified_mesh(original_points, nested_faces, tg_red=0.5):
    debug("Simplifying the city mesh...")
    simplified_points, simplified_faces, mapping = fast_simplification.simplify(
        original_points,
        nested_faces,
        target_reduction=tg_red,
        return_collapses=True
    )
    flat_simplified_faces = []
    for face in simplified_faces:
        flat_simplified_faces.append(len(face))
        flat_simplified_faces.extend(face)
    simplified_mesh = pv.PolyData(simplified_points, flat_simplified_faces)
    return simplified_mesh, simplified_points, simplified_faces, mapping

def get_graph_stats(points, faces, mesh, params, algo):
    stats = {
        "n_nodes": len(points),
        "dijkstra": [],
        "astar": [],
        "rrt": [],
        "rrtdist": 0,
        "rrttime": 0
    }
    start = params["start"]
    goal = params["goal"]
    debug("Computing paths...")
    
    try:
        rrt = RRT(start=start, goal=goal, mesh=mesh, step_size=params["step_size"], max_iter=params["max_iter"], goal_sample_rate=0.3, search_radius=params["step_size"])
        dijgraph = LazyGraph(start=start, goal=goal, mesh=mesh, step_size=params["step_size"], max_iter=params["max_iter"], search_radius=params["step_size"])
        agraph = LazyGraph(start=start, goal=goal, mesh=mesh, step_size=params["step_size"], max_iter=params["max_iter"], search_radius=params["step_size"])

        threads = []

        def run_rrt():
            path, dist = rrt.build_rrt()
            stats["rrttime"] = rrt.rtime
            stats["rrtdist"] = dist
            stats["rrt"] = path

        def run_dijkstra():
            path, dist = dijgraph.dijkstra()
            stats["dijtime"] = dijgraph.rtime
            stats["dijdist"] = dist
            stats["dijkstra"] = path

        def run_astar():
            path, dist = agraph.a_star()
            stats["astartime"] = agraph.rtime
            stats["astardist"] = dist
            stats["astar"] = path

        if algo in ["all", "rrt"]:
            threads.append(threading.Thread(target=run_rrt))
        if algo in ["all", "dijkstra"]:
            threads.append(threading.Thread(target=run_dijkstra))
        if algo in ["all", "astar"]:
            threads.append(threading.Thread(target=run_astar))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    except nx.NetworkXNoPath as e:
        debug(e)
        stats["dijkstra"].append(None)
        stats["astar"].append(None)
    return stats

def visualize(mesh, simplified_mesh, orig_paths, simpl_paths):
    debug("Visualizing paths...")
    plotter = pv.Plotter(shape=(1, 2))

    colors = ["red", "black", "orange", "purple"]

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color="blue", show_edges=True)
    if orig_paths and len(orig_paths) > 0:
        try:
            orig_paths = np.array(orig_paths)
            if orig_paths.ndim == 2 and orig_paths.shape[1] == 3:
                lines = pv.line_segments_from_points(orig_paths)
                plotter.add_mesh(lines, color=colors[0])
            else:
                debug("Original paths are not valid for visualization.")
        except Exception as e:
            debug(f"Error visualizing original paths: {e}")
    else:
        debug("No valid path found for the original mesh.")
        plotter.add_text("No valid path found for original mesh", font_size=10)

    plotter.subplot(0, 1)
    plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
    if simpl_paths and len(simpl_paths) > 0:
        try:
            simpl_paths = np.array(simpl_paths)
            if simpl_paths.ndim == 2 and simpl_paths.shape[1] == 3:
                lines = pv.line_segments_from_points(simpl_paths)
                plotter.add_mesh(lines, color=colors[0])
            else:
                debug("Simplified paths are not valid for visualization.")
        except Exception as e:
            debug(f"Error visualizing simplified paths: {e}")
    else:
        debug("No valid path found for the simplified mesh.")
        plotter.add_text("No valid path found for simplified mesh", font_size=10)

    plotter.link_views()
    plotter.show()


def get_mesh_stats(dataset_path, params, algo):
    original_data_path = dataset_path + '_original.json'
    simplified_data_path = dataset_path + '_simplified.json'

    original_data = load_mesh_data(original_data_path)
    simplified_data = load_mesh_data(simplified_data_path)

    if original_data and simplified_data:
        debug("Using cached data for mesh stats...")
        original_mesh, original_points, nested_faces = load_mesh(dataset_path)
        simplified_mesh, simplified_points, simplified_faces, _ = get_simplified_mesh(original_points, nested_faces)
    else:
        original_mesh, original_points, nested_faces = load_mesh(dataset_path)
        simplified_mesh, simplified_points, simplified_faces, mapping = get_simplified_mesh(original_points, nested_faces)
        save_mesh_data(original_data_path, {
            "points": original_points,
            "faces": nested_faces
        })
        save_mesh_data(simplified_data_path, {
            "points": simplified_points,
            "faces": simplified_faces
        })

    debug("Creating graphs for pathfinding...")

    test_results = params | {"file": dataset_path, "original": None, "simplified": None}
    def run_original():
        original_stats = get_graph_stats(original_points, nested_faces, original_mesh, params, algo)
        test_results["original"] = original_stats

    def run_simplified():
        simplified_stats = get_graph_stats(simplified_points, simplified_faces, simplified_mesh, params, algo)
        test_results["simplified"] = simplified_stats

    orig_th = threading.Thread(target=run_original)
    simpl_th = threading.Thread(target=run_simplified)

    orig_th.start()
    simpl_th.start()
    
    orig_th.join()
    simpl_th.join()
 
    output_path = f"{dataset_path}_{params['test_num']}.json"
    with open(output_path, 'w') as file:
        json.dump(test_results, file, indent=4)
    debug(f"Results saved to {output_path}")

    if params.get("visualize", False):
        visualize(original_mesh, simplified_mesh, original_stats.get("astar", []), simplified_stats.get("astar", []))

def main():
    parser = argparse.ArgumentParser(description="Mesh simplification and pathfinding")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset STL file")
    parser.add_argument("--algo", type=str, choices=["dijkstra", "astar", "rrt", "all"], default="all", help="Algorithm to run")
    parser.add_argument("--visualize", action="store_true", help="Visualize paths")
    parser.add_argument("--test-cases", type=str, help="Path to the JSON file containing start/goal test cases", required=True)
    args = parser.parse_args()

    # Load test cases from the provided JSON file
    try:
        with open(args.test_cases, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading test cases: {e}")
        sys.exit(1)

    # Loop through all test cases
    for idx, case in enumerate(test_cases):
        print(f"\nRunning test case {idx + 1}/{len(test_cases)}:")
        params = {
            "start": case.get("start"),
            "goal": case.get("goal"),
            "max_iter": case.get("max_iter", 1000),  # Default value if not specified
            "step_size": case.get("step_size", 5),  # Default value if not specified
            "test_num": idx + 1,
            "visualize": args.visualize
        }
        print(f"Start: {params['start']}, Goal: {params['goal']}")
        get_mesh_stats(args.dataset, params, args.algo)

if __name__ == "__main__":
    main()
