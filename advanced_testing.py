import pyvista as pv
import fast_simplification

# Load a larger dataset: Nefertiti
# original_mesh = pv.examples.download_nefertiti()
# original_mesh = pv.examples.download_bunny()
# original_mesh = pv.examples.download_cow()
original_mesh = pv.examples.download_teapot()
# original_mesh = pv.examples.download_dragon()



# Extract points and faces for simplification
original_points = original_mesh.points.tolist()  # Convert to a list for compatibility
original_faces = original_mesh.faces.reshape(-1, 4)[:, 1:]  # Extract face indices (triangles only)
nested_faces = original_faces.tolist()  # Convert to a nested list for simplification

# Simplify the mesh using fast-simplification
simplified_points, simplified_faces = fast_simplification.simplify(
    original_points,
    nested_faces,
    target_reduction=1.0  # Adjust target reduction as needed
)

# Reformat simplified faces for PyVista (flatten with vertex count)
flat_simplified_faces = []
for face in simplified_faces:
    flat_simplified_faces.append(len(face))  # Number of vertices in the face
    flat_simplified_faces.extend(face)       # Indices of the vertices

# Create the simplified PyVista mesh
simplified_mesh = pv.PolyData(simplified_points, flat_simplified_faces)

# Visualize the original and simplified meshes side by side
plotter = pv.Plotter(shape=(1, 2))

# Original mesh
plotter.subplot(0, 0)
plotter.add_mesh(original_mesh, color="blue", show_edges=True)
plotter.add_text("Original Mesh", font_size=10)

# Simplified mesh
plotter.subplot(0, 1)
plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
plotter.add_text("Simplified Mesh", font_size=10)

# Show the plot
plotter.link_views()  # Link camera views for side-by-side comparison
plotter.show()
