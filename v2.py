import pyvista as pv
import numpy as np

# Hardcoded dataset path and stats
dataset_path = "dataset/city.stl"
stats = {
    "n_nodes": 1916325,
    "dijkstra": [(216, 678, 22), (218.7, 672.6, 27.2), (218.7, 672.6, 27.2), (221.4, 667.2, 32.5), (221.4, 667.2, 32.5), (224.2, 661.8, 37.7), (224.2, 661.8, 37.7), (227.0, 656.4, 43.0), (227.0, 656.4, 43.0), (229.7, 651.0, 48.3), (229.7, 651.0, 48.3), (232, 646.6, 52.6)],
    "astar": [(216, 678, 22), (216, 678, 30), (216, 678, 30), (219.1, 672.0, 34.3), (219.1, 672.0, 34.3), (219.1, 664.0, 34.3), (219.1, 664.0, 34.3), (222.7, 659.1, 39.5), (222.7, 659.1, 39.5), (226.4, 654.2, 44.6), (226.4, 654.2, 44.6), (230.0, 649.3, 49.8), (230.0, 649.3, 49.8), (232, 646.6, 52.6)],
    "rrt": [(216, 678, 22), (218.74251574887998, 672.617812842823, 27.24506136973295), (218.74251574887998, 672.617812842823, 27.24506136973295), (221.0831168705111, 665.0491915791581, 28.35751746172897), (221.0831168705111, 665.0491915791581, 28.35751746172897), (220.8032324089286, 657.835288195307, 31.804438211706717), (220.8032324089286, 657.835288195307, 31.804438211706717), (224.22804835243275, 654.3986897459621, 38.165289322295614), (224.22804835243275, 654.3986897459621, 38.165289322295614), (225.48962398536892, 646.5041913482419, 37.87319007413683), (225.48962398536892, 646.5041913482419, 37.87319007413683), (228.72419972220115, 646.5517923377483, 45.18996900673376), (228.72419972220115, 646.5517923377483, 45.18996900673376), (231.95877545903338, 646.5993933272548, 52.506747939330694), (231.95877545903338, 646.5993933272548, 52.506747939330694), (232.0, 646.6, 52.6)],
}

def load_mesh(dataset_path):
    """Load the STL mesh dataset."""
    print("Loading STL dataset...")
    return pv.read(dataset_path)

def visualize_paths(mesh, stats):
    """Visualize the mesh and paths."""
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="blue", show_edges=True, opacity=0.5)

    # Colors for different algorithms
    algo_colors = {
        "dijkstra": "red",
        "astar": "green",
        "rrt": "orange"
    }

    # Visualize paths from stats
    for algo, color in algo_colors.items():
        if algo in stats and stats[algo]:
            path = stats[algo]
            if len(path) >= 2:
                try:
                    path_segments = []
                    for i in range(len(path) - 1):
                        path_segments.extend([path[i], path[i + 1]])
                    path_segments = np.array(path_segments)
                    plotter.add_lines(path_segments, color=color, width=2, label=f"{algo.capitalize()} Path")
                except Exception as e:
                    print(f"Error visualizing {algo} path: {e}")

    plotter.add_legend()
    plotter.show()

# Load dataset and visualize
mesh = load_mesh(dataset_path)
visualize_paths(mesh, stats)


