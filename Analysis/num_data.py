# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Data
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

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def compute_stats(arr):
#     """Compute basic statistics for a given array."""
#     return {
#         "mean": np.nanmean(arr),
#         "median": np.nanmedian(arr),
#         "std": np.nanstd(arr),
#         "min": np.nanmin(arr),
#         "max": np.nanmax(arr)
#     }

# def summarize_data(data):
#     """Compute statistics for all metrics in the dataset."""
#     metrics = {
#         "rrtdist": {
#             "original": np.array([entry["original"]["rrtdist"] for entry in data]),
#             "simplified": np.array([entry["simplified"]["rrtdist"] for entry in data])
#         },
#         "rrttime": {
#             "original": np.array([entry["original"]["rrttime"] for entry in data]),
#             "simplified": np.array([entry["simplified"]["rrttime"] for entry in data])
#         },
#         "astardist": {
#             "original": np.array([entry["original"]["astardist"] if entry["original"]["astardist"] is not None else np.nan for entry in data]),
#             "simplified": np.array([entry["simplified"]["astardist"] if entry["simplified"]["astardist"] is not None else np.nan for entry in data])
#         },
#         "astartime": {
#             "original": np.array([entry["original"]["astartime"] if entry["original"]["astartime"] is not None else np.nan for entry in data]),
#             "simplified": np.array([entry["simplified"]["astartime"] if entry["simplified"]["astartime"] is not None else np.nan for entry in data])
#         },
#     }

#     stats = {}
#     for metric, datasets in metrics.items():
#         stats[metric] = {
#             "original": compute_stats(datasets["original"]),
#             "simplified": compute_stats(datasets["simplified"])
#         }
#     return stats

# def plot_stats(stats):
#     """Plot the statistics in a bar chart."""
#     metrics = stats.keys()
#     categories = ["mean", "median", "std", "min", "max"]

#     # Create a plot for each metric
#     for metric in metrics:
#         original_values = [stats[metric]["original"][cat] for cat in categories]
#         simplified_values = [stats[metric]["simplified"][cat] for cat in categories]

#         x = np.arange(len(categories))  # Label locations
#         width = 0.35  # Bar width

#         fig, ax = plt.subplots()
#         ax.bar(x - width/2, original_values, width, label="Original")
#         ax.bar(x + width/2, simplified_values, width, label="Simplified")

#         ax.set_xlabel("Statistic")
#         ax.set_ylabel(metric.capitalize())
#         ax.set_title(f"{metric.capitalize()} Statistics")
#         ax.set_xticks(x)
#         ax.set_xticklabels(categories)
#         ax.legend()

#         plt.show()

# def visualize_metrics(data):
#     """Generate numerical summaries and visualizations."""
#     stats = summarize_data(data)
#     df_stats = pd.DataFrame.from_dict(
#         {(metric, version): stats[metric][version] for metric in stats for version in stats[metric]},
#         orient="index"
#     ).unstack(level=0)

#     print("\nNumerical Summary:")
#     print(df_stats.round(3))  # Display with rounded values

#     plot_stats(stats)  # Generate bar plots for each metric

# # Main execution
# visualize_metrics(data)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
# data = [
#     # Paste your data here
# ]

def compute_stats(arr):
    """Compute basic statistics for a given array."""
    return {
        "mean": np.nanmean(arr),
        "median": np.nanmedian(arr),
        "std": np.nanstd(arr),
        "min": np.nanmin(arr),
        "max": np.nanmax(arr)
    }

def summarize_data(data):
    """Compute statistics for all metrics in the dataset."""
    metrics = {
        "rrtdist": {
            "original": np.array([entry["original"]["rrtdist"] for entry in data]),
            "simplified": np.array([entry["simplified"]["rrtdist"] for entry in data])
        },
        "rrttime": {
            "original": np.array([entry["original"]["rrttime"] for entry in data]),
            "simplified": np.array([entry["simplified"]["rrttime"] for entry in data])
        },
        "astardist": {
            "original": np.array([entry["original"]["astardist"] if entry["original"]["astardist"] is not None else np.nan for entry in data]),
            "simplified": np.array([entry["simplified"]["astardist"] if entry["simplified"]["astardist"] is not None else np.nan for entry in data])
        },
        "astartime": {
            "original": np.array([entry["original"]["astartime"] if entry["original"]["astartime"] is not None else np.nan for entry in data]),
            "simplified": np.array([entry["simplified"]["astartime"] if entry["simplified"]["astartime"] is not None else np.nan for entry in data])
        },
    }

    stats = {}
    for metric, datasets in metrics.items():
        stats[metric] = {
            "original": compute_stats(datasets["original"]),
            "simplified": compute_stats(datasets["simplified"])
        }
    return stats

def plot_stats(stats):
    """Plot the statistics in a bar chart."""
    metrics = stats.keys()
    categories = ["mean", "median", "std", "min", "max"]

    # Create a plot for each metric
    for metric in metrics:
        original_values = [stats[metric]["original"][cat] for cat in categories]
        simplified_values = [stats[metric]["simplified"][cat] for cat in categories]

        x = np.arange(len(categories))  # Label locations
        width = 0.35  # Bar width

        fig, ax = plt.subplots()
        ax.bar(x - width/2, original_values, width, label="Original")
        ax.bar(x + width/2, simplified_values, width, label="Simplified")

        ax.set_xlabel("Statistic")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Statistics")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.show()

def save_summary(stats, filename="summary.csv"):
    """Save the numerical summary to a CSV file."""
    df_stats = pd.DataFrame.from_dict(
        {(metric, version): stats[metric][version] for metric in stats for version in stats[metric]},
        orient="index"
    ).unstack(level=0)

    df_stats.to_csv(filename)
    print(f"\nNumerical summary saved to {filename}")

    # Print the full DataFrame without truncation
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("\nNumerical Summary:")
    print(df_stats.round(3))

def visualize_metrics(data):
    """Generate numerical summaries and visualizations."""
    stats = summarize_data(data)
    save_summary(stats, "summary.csv")
    plot_stats(stats)  # Generate bar plots for each metric

# Main execution
visualize_metrics(data)
