import fast_simplification

# Define points and faces of a simple mesh
points = [
    [0.5, -0.5, 0.0],
    [0.0, -0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [0.5,  0.0, 0.0],
    [0.0,  0.0, 0.0],
    [-0.5,  0.0, 0.0],
    [0.5,  0.5, 0.0],
    [0.0,  0.5, 0.0],
    [-0.5,  0.5, 0.0],
]

faces = [
    [0, 1, 3],
    [4, 3, 1],
    [1, 2, 4],
    [5, 4, 2],
    [3, 4, 6],
    [7, 6, 4],
    [4, 5, 7],
    [8, 7, 5],
]

# Simplify the mesh
points_out, faces_out = fast_simplification.simplify(points, faces, 0.5)

print("Simplified Points:", points_out)
print("Simplified Faces:", faces_out)
