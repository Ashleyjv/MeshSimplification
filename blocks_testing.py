import pyvista as pv
import fast_simplification
import sys

DEBUG = True

def debug(*args, sep=' ', end='\n', file=None, flush=False):
    # Check the environment variable
    if DEBUG:
        # If file is None, default to sys.stdout, just like print
        if file is None:
            file = sys.stdout
        print(*args, sep=sep, end=end, file=file, flush=flush)

# Path to the dataset
dataset_path = "dataset/city.stl"

# Load the STL file
print("Loading city STL dataset...")
mesh = pv.read(dataset_path)

# Extract points and faces for simplification
original_points = mesh.points.tolist()
original_faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Extract triangle indices
nested_faces = original_faces.tolist()

# Simplify the mesh
print("Simplifying the city mesh...")
simplified_points, simplified_faces = fast_simplification.simplify(
    original_points,
    nested_faces,
    target_reduction=1.0  # Adjust as needed
)

# Reformat simplified faces for PyVista
flat_simplified_faces = []
for face in simplified_faces:
    flat_simplified_faces.append(len(face))  # Number of vertices in the face
    flat_simplified_faces.extend(face)       # Indices of the vertices

# Create a simplified PyVista mesh
simplified_mesh = pv.PolyData(simplified_points, flat_simplified_faces)

# Visualize the original and simplified meshes side by side
print("Visualizing the results...")
plotter = pv.Plotter(shape=(1, 2))

# Original mesh
plotter.subplot(0, 0)
plotter.add_mesh(mesh, color="blue", show_edges=True)
plotter.add_text("Original Mesh", font_size=10)

# Simplified mesh
plotter.subplot(0, 1)
plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
plotter.add_text("Simplified Mesh", font_size=10)

# Show the plot
plotter.link_views()
plotter.show()
