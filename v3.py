import numpy as np
import pyvista as pv
import json

# Define debug function for logging
def debug(message):
    print(f"[DEBUG] {message}")


# Load JSON data
def load_data(json_file):
    """Loads path and mesh data from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


# Extract paths from JSON data
def extract_paths(data, key):
    """Extracts paths from the JSON data based on the key ('original' or 'simplified')."""
    paths = data.get(key, {}).get('astar', [])  # Adjust 'astar' if you want other paths
    return paths


# Visualization function
def visualize(mesh_file, json_file):
    """
    Visualizes the original and simplified meshes along with their respective paths.

    Args:
        mesh_file: Path to the original STL file.
        json_file: Path to the JSON file containing paths and simplified mesh info.
    """
    debug("Loading mesh and path data...")
    # Load the meshes
    mesh = pv.read(mesh_file)
    simplified_mesh = mesh  # Replace with actual simplified mesh if available

    # Load JSON data
    data = load_data(json_file)

    # Extract paths
    orig_paths = extract_paths(data, 'original')
    simpl_paths = extract_paths(data, 'simplified')

    debug("Visualizing paths...")
    plotter = pv.Plotter(shape=(1, 2))  # Create a subplot with 1 row and 2 columns

    # Define colors for paths
    colors = ["red", "black", "orange", "purple"]

    # Plot the original mesh and paths
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color="blue", show_edges=True)
    plotter.add_text("Original Mesh", font_size=10)
    if orig_paths and len(orig_paths) > 0:
        try:
            orig_paths = np.array(orig_paths)
            if orig_paths.ndim == 2 and orig_paths.shape[1] == 3:
                lines = pv.lines_from_points(orig_paths)
                plotter.add_mesh(lines, color=colors[0], line_width=2, label="Original Path")
            else:
                debug("Original paths are not valid for visualization.")
        except Exception as e:
            debug(f"Error visualizing original paths: {e}")
    else:
        debug("No valid path found for the original mesh.")
        plotter.add_text("No valid path found for original mesh", font_size=10)

    # Plot the simplified mesh and paths
    plotter.subplot(0, 1)
    plotter.add_mesh(simplified_mesh, color="green", show_edges=True)
    plotter.add_text("Simplified Mesh", font_size=10)
    if simpl_paths and len(simpl_paths) > 0:
        try:
            simpl_paths = np.array(simpl_paths)
            if simpl_paths.ndim == 2 and simpl_paths.shape[1] == 3:
                lines = pv.lines_from_points(simpl_paths)
                plotter.add_mesh(lines, color=colors[1], line_width=2, label="Simplified Path")
            else:
                debug("Simplified paths are not valid for visualization.")
        except Exception as e:
            debug(f"Error visualizing simplified paths: {e}")
    else:
        debug("No valid path found for the simplified mesh.")
        plotter.add_text("No valid path found for simplified mesh", font_size=10)

    # Link the two subplots for synchronized camera movement
    plotter.link_views()
    plotter.show()


# Example usage
if __name__ == "__main__":
    # Provide paths to the STL file and JSON file
    original_mesh_path = "/Users/ashleyjvarghese/Desktop/MeshSimplification/MeshSimplification/dataset/city.stl"  # Replace with your STL file path
    json_path = "/Users/ashleyjvarghese/Desktop/MeshSimplification/MeshSimplification/dataset/city.stl_7.json"  # Replace with your JSON file path

    # Visualize the meshes and paths
    visualize(original_mesh_path, json_path)
