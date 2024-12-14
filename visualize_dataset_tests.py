import argparse
import pyvista as pv
import numpy as np
import random
import json
import os

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    print(*args, sep=sep, end=end, file=file, flush=flush)

def load_mesh(dataset_path):
    """Load the dataset mesh from an STL file."""
    debug("Loading city STL dataset...")
    mesh = pv.read(dataset_path)
    return mesh

def is_collision_free(mesh, point):
    """Check if a point is in free space by ray tracing."""
    bounds = mesh.bounds
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    ray_direction = np.array(point) - np.array(center)
    ray_direction /= np.linalg.norm(ray_direction)  # Normalize the direction

    # Cast a ray and count intersections
    intersections, _ = mesh.ray_trace(center, point)
    return len(intersections) == 0

def generate_random_points(mesh, num_points):
    """Generate random points within the mesh bounds."""
    bounds = mesh.bounds
    points = []
    while len(points) < num_points:
        point = [
            random.uniform(bounds[0], bounds[1]),
            random.uniform(bounds[2], bounds[3]),
            random.uniform(bounds[4], bounds[5])
        ]
        if is_collision_free(mesh, point):
            points.append(point)
    return points

def save_test_cases(test_cases, output_path):
    """Save test cases to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(test_cases, file, indent=4)

def visualize_mesh_and_points(mesh, points=None, start=None, goal=None):
    """Visualize the mesh and test points, highlighting start and goal points."""
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue", opacity=0.5)

    if points is not None and len(points) > 0:
        points_array = np.array(points)
        plotter.add_points(points_array, color="green", point_size=5, label="Test Points")

    if start is not None:
        plotter.add_points(np.array([start]), color="red", point_size=10, label="Start Point")
    if goal is not None:
        plotter.add_points(np.array([goal]), color="blue", point_size=10, label="Goal Point")

    plotter.add_legend()
    plotter.show()

def generate_test_cases(mesh, num_cases, visualize=False):
    """Generate test cases with random start and goal points in free space."""
    test_cases = []
    for _ in range(num_cases):
        points = generate_random_points(mesh, 2)  # Generate start and goal
        start, goal = points[0], points[1]
        test_case = {"start": start, "goal": goal}
        test_cases.append(test_case)

        if visualize:
            visualize_mesh_and_points(mesh, start=start, goal=goal)

    return test_cases

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize test cases for a mesh dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset STL file.")
    parser.add_argument("--output", type=str, default="test_cases.json", help="Path to save the generated test cases.")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of test cases to generate.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the mesh and test cases.")
    args = parser.parse_args()

    # Load the dataset mesh
    mesh = load_mesh(args.dataset)

    # Generate test cases
    debug("Generating test cases...")
    test_cases = generate_test_cases(mesh, args.num_cases, visualize=args.visualize)

    # Save test cases
    save_test_cases(test_cases, args.output)
    debug(f"Test cases saved to {args.output}")

    if args.visualize:
        debug("Visualizing the final set of test cases...")
        all_points = [case["start"] for case in test_cases] + [case["goal"] for case in test_cases]
        visualize_mesh_and_points(mesh, points=all_points)

if __name__ == "__main__":
    main()
#python visualize_dataset_tests.py --dataset dataset/city.stl --output test_cases.json --num_cases 10 --visualize
