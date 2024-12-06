import pyvista as pv
import fast_simplification
import networkx as nx
import numpy as np

# Path to the dataset
dataset_path = "dataset/city.stl"

# Load the STL file
print("Loading city STL dataset...")
mesh = pv.read(dataset_path)

# Extract points and faces for graph creation and simplification
original_points = mesh.points.tolist()
original_faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Extract triangle indices
nested_faces = original_faces.tolist()

# Simplify the mesh
print("Simplifying the city mesh...")
simplified_points, simplified_faces = fast_simplification.simplify(
    original_points,
    nested_faces,
    target_reduction=0.75  # Adjust as needed
)

# Reformat simplified faces for PyVista
flat_simplified_faces = []
for face in simplified_faces:
    flat_simplified_faces.append(len(face))  # Number of vertices in the face
    flat_simplified_faces.extend(face)

# Create simplified PyVista mesh
simplified_mesh = pv.PolyData(simplified_points, flat_simplified_faces)

# Function to create a graph from a mesh
def create_graph(points, faces, max_height=None):
    graph = nx.Graph()
    for i, point in enumerate(points):
        graph.add_node(i, pos=tuple(point))
    for face in faces:
        for j in range(len(face)):
            # Connect vertices if the height constraint is met
            if max_height is None or abs(points[face[j]][2] - points[face[(j + 1) % len(face)]][2]) <= max_height:
                graph.add_edge(face[j], face[(j + 1) % len(face)])  # Connect vertices cyclically
    return graph

# Define a height constraint (e.g., max height difference of 2 units)
max_height_constraint = 2.0

# Create graphs
print("Creating graphs for pathfinding...")
graph_original = create_graph(original_points, nested_faces, max_height=max_height_constraint)
graph_simplified = create_graph(simplified_points, simplified_faces, max_height=max_height_constraint)

# Define source and destination
src_idx = 0
dst_idx = min(len(original_points) - 1, 500)  # Use a closer node to ensure connectivity

# Validate path existence
if not nx.has_path(graph_original, src_idx, dst_idx):
    raise ValueError(f"No path found between {src_idx} and {dst_idx} in the original graph.")

if not nx.has_path(graph_simplified, src_idx, dst_idx):
    raise ValueError(f"No path found between {src_idx} and {dst_idx} in the simplified graph.")

# Find shortest paths
print("Computing paths...")
path_original = nx.shortest_path(graph_original, source=src_idx, target=dst_idx, weight=None)
path_simplified = nx.shortest_path(graph_simplified, source=src_idx, target=dst_idx, weight=None)

# Helper function to extract path points
def get_path_points(path, points):
    return np.array([points[idx] for idx in path])

# Extract path points
path_points_original = get_path_points(path_original, original_points)
path_points_simplified = get_path_points(path_simplified, simplified_points)

# Visualize
print("Visualizing paths...")
plotter = pv.Plotter(shape=(1, 2))

# Original mesh with path
plotter.subplot(0, 0)
plotter.add_mesh(mesh, color="blue", show_edges=True)
plotter.add_lines(path_points_original, color="red", width=5)
plotter.add_text("Original Mesh with Path", font_size=10)

# Simplified mesh with path
plotter.subplot(0, 1)
plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
plotter.add_lines(path_points_simplified, color="red", width=5)
plotter.add_text("Simplified Mesh with Path", font_size=10)

# Show the plot
plotter.link_views()
plotter.show()
