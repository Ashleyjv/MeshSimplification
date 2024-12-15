# Mesh Simplification

Pathfinding in 3D environments is a critical challenge in robotics, often constrained by the computational complexity of navigating detailed maps. This study explores the impact of mesh simplification on 3D pathfinding algorithms by analyzing changes in computation time and path quality. Starting with an irregular 3D map dataset, we implement popular search algorithms, including A*, Dijkstra, and RRT, to evaluate their performance on both the original and simplified meshes. Mesh simplification is performed using Blender and Python-based APIs, reducing the computational burden while preserving navigational features. Results demonstrate significant reductions in computation time and resource usage post-simplification, with minimal compromise in path quality. Comparative analysis highlights trade-offs between path-finding accuracy and processing efficiency, providing insights into optimizing navigation in robotics applications. This work emphasizes the potential of mesh simplification as a preprocessing step to enhance the performance of 3D navigation systems, offering a scalable approach for real-time robotics tasks.

---

## Getting Started

### Dependencies

Install the required Python libraries:
```bash
pip3 install -r requirements_test.txt
pip3 install -r requirements_docs.txt

```

### Installing

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mesh-simplification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mesh-simplification
   ```
3. Install the dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

### Executing Program

#### Usage
To run the program, use the following command:
```bash
python3 main.py --dataset <path_to_dataset_stl_file> \
                --algo <algorithm_choice> \
                --test-cases <path_to_test_cases_json> \
                [--visualize]
```

#### Arguments:
- `--dataset`: Path to the dataset STL file. Place your `.stl` file in the `dataset` directory.
- `--algo`: Algorithm to use. Choices are:
  - `dijkstra`
  - `astar`
  - `rrt`
  - `all` (default, runs all algorithms).
- `--test-cases`: Path to a JSON file containing test cases. The file should be structured like:
  ```json
  [
      {"start": [0, 0, 0], "goal": [10, 10, 10], "max_iter": 50000, "step_size": 5},
      {"start": [1, 1, 1], "goal": [20, 20, 20]}
  ]
  ```
- `--visualize`: Optional. Enables path visualization in PyVista.

#### Example Command:
```bash
python3 run.py --dataset data/city.stl \
                --algo all \
                --test-cases data/test_cases.json \
                --visualize
```

---

### Generating Test Cases

You can use the `visualize_dataset_tests.py` file to generate random test cases for your dataset. This script creates a JSON file with random `start` and `goal` points that can be fed into `main.py`.

#### Usage:
```bash
python3 visualize_dataset_tests.py --dataset dataset/city.stl \
                                   --output test_cases.json \
                                   --num_cases 10 \
                                   --visualize
```

#### Arguments:
- `--dataset`: Path to the dataset STL file. Place your `.stl` file in the `dataset` directory.
- `--output`: Path to save the generated test cases JSON file.
- `--num_cases`: Number of test cases to generate (default: 10).
- `--visualize`: Optional. Visualizes the mesh and test points using PyVista.

#### Example Command:
```bash
python3 visualize_dataset_tests.py --dataset dataset/city.stl \
                                   --output dataset/test_cases.json \
                                   --num_cases 10 \
                                   --visualize
```
---
## Analysis:

Analysis of generated data is performed using the num_data.py and vis_data.py scripts located in the Analysis directory.

num_data.py: calculates and output summary statistics and visualizations. (Update data field required)

vis_data.py: Generate visualizations for statistics per test, which are automatically saved in the output directory. (Update data field required)

Ensure you update the respective data fields in these scripts before running them for accurate results.


---

## Help

If you encounter issues, you can use the following command to view help information:
```bash
python3 run.py --help
```

---
