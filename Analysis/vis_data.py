import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    {
        "file": "dataset/city.stl_1.json",
        "original": {
            "rrtdist": 94.44372706658652,
            "rrttime": 4.370496511459351,
            "astardist": 95.50506562615962,
            "astartime": 5.493967533111572
        },
        "simplified": {
            "rrtdist": 117.51793097464892,
            "rrttime": 2.659456968307495,
            "astardist": 95.50506562615962,
            "astartime": 3.098208427429199
        }
    },
    {
        "file": "dataset/city.stl_2.json",
        "original": {
            "rrtdist": 344.3263133806799,
            "rrttime": 20.810006380081177,
            "astardist": None,
            "astartime": 0.007855653762817383
        },
        "simplified": {
            "rrtdist": 286.4221872763664,
            "rrttime": 7.8169591426849365,
            "astardist": None,
            "astartime": 0.005873680114746094
        }
    },
    {
        "file": "dataset/city.stl_3.json",
        "original": {
            "rrtdist": 124.14784535209301,
            "rrttime": 4.7444422245025635,
            "astardist": 115.63016378270152,
            "astartime": 5.085635185241699
        },
        "simplified": {
            "rrtdist": 203.1057399963479,
            "rrttime": 18.477664709091187,
            "astardist": 115.63016378270152,
            "astartime": 5.039812326431274
        }
    },
    {
        "file": "dataset/city.stl_4.json",
        "original": {
            "rrtdist": 130.5605117472645,
            "rrttime": 4.417320489883423,
            "astardist": None,
            "astartime": 0.9473111629486084
        },
        "simplified": {
            "rrtdist": 118.03968323264041,
            "rrttime": 1.4874048233032227,
            "astardist": None,
            "astartime": 0.00832676887512207
        }
    },
    {
        "file": "dataset/city.stl_5.json",
        "original": {
            "rrtdist": 96.69342861965495,
            "rrttime": 5.003108739852905,
            "astardist": None,
            "astartime": 0.13604211807250977
        },
        "simplified": {
            "rrtdist": 107.88882579449988,
            "rrttime": 2.7837326526641846,
            "astardist": None,
            "astartime": 0.010469675064086914
        }
    },
    {
        "file": "dataset/city.stl_6.json",
        "original": {
            "rrtdist": 99.9915411220451,
            "rrttime": 5.939651250839233,
            "astardist": None,
            "astartime": 0.11803770065307617
        },
        "simplified": {
            "rrtdist": 98.41057807308577,
            "rrttime": 5.7379069328308105,
            "astardist": None,
            "astartime": 0.305225133895874
        }
    },
    {
        "file": "dataset/city.stl_7.json",
        "original": {
            "rrtdist": 13.77443306379469,
            "rrttime": 0.3578970432281494,
            "astardist": 12.215182013362611,
            "astartime": 0.13297533988952637
        },
        "simplified": {
            "rrtdist": 12.425764512385072,
            "rrttime": 0.11008048057556152,
            "astardist": 12.215182013362611,
            "astartime": 0.1217646598815918
        }
    },
    {
        "file": "dataset/city.stl_8.json",
        "original": {
            "rrtdist": 198.32753054268633,
            "rrttime": 6.165860176086426,
            "astardist": None,
            "astartime": 0.10512781143188477
        },
        "simplified": {
            "rrtdist": 199.00048820699237,
            "rrttime": 2.561002731323242,
            "astardist": None,
            "astartime": 0.04426264762878418
        }
    },
    {
        "file": "dataset/city.stl_9.json",
        "original": {
            "rrtdist": 122.2040637240707,
            "rrttime": 3.2805631160736084,
            "astardist": None,
            "astartime": 0.24216628074645996
        },
        "simplified": {
            "rrtdist": 155.4882588837902,
            "rrttime": 2.673408269882202,
            "astardist": None,
            "astartime": 0.013110160827636719
        }
    },
    {
        "file": "dataset/city.stl_10.json",
        "original": {
            "rrtdist": 192.67423466025707,
            "rrttime": 8.70898175239563,
            "astardist": None,
            "astartime": 0.11570858955383301
        },
        "simplified": {
            "rrtdist": 199.32825174192925,
            "rrttime": 4.123379945755005,
            "astardist": None,
            "astartime": 0.057498931884765625
        }
    }
]

test_cases = [entry["file"].split("_")[-1].split(".")[0] for entry in data]
original_rrt_dist = [entry["original"]["rrtdist"] for entry in data]
simplified_rrt_dist = [entry["simplified"]["rrtdist"] for entry in data]

original_rrt_time = [entry["original"]["rrttime"] for entry in data]
simplified_rrt_time = [entry["simplified"]["rrttime"] for entry in data]

original_astar_dist = [entry["original"]["astardist"] if entry["original"]["astardist"] is not None else np.nan for entry in data]
simplified_astar_dist = [entry["simplified"]["astardist"] if entry["simplified"]["astardist"] is not None else np.nan for entry in data]

original_astar_time = [entry["original"]["astartime"] if entry["original"]["astartime"] is not None else np.nan for entry in data]
simplified_astar_time = [entry["simplified"]["astartime"] if entry["simplified"]["astartime"] is not None else np.nan for entry in data]

# Visualization function
def plot_metric(metric_original, metric_simplified, metric_name, ylabel):
    x = np.arange(len(test_cases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_original = ax.bar(x - width/2, metric_original, width, label='Original', color='blue', alpha=0.7)
    bar_simplified = ax.bar(x + width/2, metric_simplified, width, label='Simplified', color='orange', alpha=0.7)

    ax.set_xlabel('Test Case')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{metric_name} Comparison (Original vs Simplified)')
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases)
    ax.legend()

    # Annotate bars
    for bar in bar_original + bar_simplified:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text slightly above the bar
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Plot each metric
plot_metric(original_rrt_dist, simplified_rrt_dist, "RRT Distance", "Distance")
plot_metric(original_rrt_time, simplified_rrt_time, "RRT Time", "Time (s)")
plot_metric(original_astar_dist, simplified_astar_dist, "A* Distance", "Distance")
plot_metric(original_astar_time, simplified_astar_time, "A* Time", "Time (s)")
