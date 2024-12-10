import pyvista as pv
import fast_simplification
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import random
# Path to the dataset
dataset_path = "dataset/city.stl"

# Load the STL file
print("Loading city STL dataset...")
mesh = pv.read(dataset_path)

# Extract points and faces for graph creation and simplification
original_points = mesh.points.tolist()
original_faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Extract triangle indices
nested_faces = original_faces.tolist()

# Simplify the mesh with node mapping
print("Simplifying the city mesh...")
simplified_points, simplified_faces, mapping = fast_simplification.simplify(
    original_points,
    nested_faces,
    target_reduction=0.5,  # Less aggressive simplification
    return_collapses=True  # Return node mapping
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
                graph.add_edge(face[j], face[(j + 1) % len(face)])
    return graph

# Define a height constraint (optional)
max_height_constraint = 120.0

# Create graphs
print("Creating graphs for pathfinding...")
graph_original = create_graph(original_points, nested_faces, max_height=max_height_constraint)
graph_simplified = create_graph(simplified_points, simplified_faces, max_height=max_height_constraint)

# Debug graph properties
print("Original Graph:")
print(f"  Nodes: {graph_original.number_of_nodes()}, Edges: {graph_original.number_of_edges()}")
print(f"  Connected Components: {nx.number_connected_components(graph_original)}")

print("Simplified Graph:")
print(f"  Nodes: {graph_simplified.number_of_nodes()}, Edges: {graph_simplified.number_of_edges()}")
print(f"  Connected Components: {nx.number_connected_components(graph_simplified)}")

# Focus on largest connected components
largest_cc_original = max(nx.connected_components(graph_original), key=len)
largest_cc_simplified = max(nx.connected_components(graph_simplified), key=len)

subgraph_original = graph_original.subgraph(largest_cc_original)
subgraph_simplified = graph_simplified.subgraph(largest_cc_simplified)

print(f"Largest CC (Original): {len(largest_cc_original)} nodes")
print(f"Largest CC (Simplified): {len(largest_cc_simplified)} nodes")

# Visualize graph connectivity
plt.figure(figsize=(10, 8))
nx.draw(subgraph_original, node_size=10, edge_color="blue", with_labels=False)
plt.title("Largest Connected Component (Original)")
plt.show()

plt.figure(figsize=(10, 8))
nx.draw(subgraph_simplified, node_size=10, edge_color="green", with_labels=False)
plt.title("Largest Connected Component (Simplified)")
plt.show()

# Create a KDTree for nearest neighbor mapping
simplified_tree = KDTree(simplified_points)

# Helper function to remap isolated nodes
def get_mapped_or_nearest(node, mapping, simplified_points):
    if node < len(mapping) and mapping[node] != -1:  # Check if node is mapped
        return mapping[node]
    # Fallback to nearest node in simplified points
    original_point = original_points[node]
    nearest_idx = simplified_tree.query(original_point)[1]
    print(f"Node {node} not mapped, using nearest node {nearest_idx}")
    return nearest_idx

# Select source and destination nodes for pathfinding
src_idx = random.choice(list(largest_cc_original))
dst_idx = random.choice(list(largest_cc_original - {src_idx}))

print(src_idx, dst_idx)
# Map source and destination nodes
simplified_src_idx = get_mapped_or_nearest(src_idx, mapping, simplified_points)
simplified_dst_idx = get_mapped_or_nearest(dst_idx, mapping, simplified_points)

# Pathfinding
print(f"Source Node: {src_idx}, Destination Node: {dst_idx}")
if nx.has_path(subgraph_original, src_idx, dst_idx):
    path_original = nx.shortest_path(subgraph_original, source=src_idx, target=dst_idx)
    print(f"Path in Original Graph: {path_original}")
else:
    print("No path found in Original Graph.")

if nx.has_path(subgraph_simplified, simplified_src_idx, simplified_dst_idx):
    path_simplified = nx.shortest_path(subgraph_simplified, source=simplified_src_idx, target=simplified_dst_idx)
    print(f"Path in Simplified Graph: {path_simplified}")
else:
    print("No path found in Simplified Graph.")

# # Visualize paths
# if 'path_original' in locals() and 'path_simplified' in locals():
#     path_points_original = np.array([original_points[node] for node in path_original])
#     path_points_simplified = np.array([simplified_points[node] for node in path_simplified])

#     # Create line segments for visualization
#     line_segments_original = np.vstack([path_points_original[:-1], path_points_original[1:]]).reshape(-1, 3)
#     line_segments_simplified = np.vstack([path_points_simplified[:-1], path_points_simplified[1:]]).reshape(-1, 3)

#     plotter = pv.Plotter(shape=(1, 2))
#     plotter.subplot(0, 0)
#     plotter.add_mesh(mesh, color="blue", show_edges=True)
#     plotter.add_lines(line_segments_original, color="red", width=5)
#     plotter.add_text("Original Mesh", font_size=10)

#     plotter.subplot(0, 1)
#     plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
#     plotter.add_lines(line_segments_simplified, color="red", width=5)
#     plotter.add_text("Simplified Mesh", font_size=10)

#     plotter.link_views()
#     plotter.show()
