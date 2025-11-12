import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data with fractional accuracies
data = [
    {"Algorithm": "ERM",       "noise": 0.1,  "acc": 0.622},
    {"Algorithm": "ERM",       "noise": 0.25, "acc": 0.549},
    {"Algorithm": "IRM",       "noise": 0.1,  "acc": 0.612},
    {"Algorithm": "IRM",       "noise": 0.25, "acc": 0.539},
    {"Algorithm": "GroupDRO",  "noise": 0.1,  "acc": 0.613},
    {"Algorithm": "GroupDRO",  "noise": 0.25, "acc": 0.541},
    {"Algorithm": "Mixup",     "noise": 0.1,  "acc": 0.639},
    {"Algorithm": "Mixup",     "noise": 0.25, "acc": 0.575},
    {"Algorithm": "VREx",      "noise": 0.1,  "acc": 0.609},
    {"Algorithm": "VREx",      "noise": 0.25, "acc": 0.530},
    {"Algorithm": "Ours",      "noise": 0.1,  "acc": 0.682},
    {"Algorithm": "Ours",      "noise": 0.25, "acc": 0.652},
]

df = pd.DataFrame(data)
# Sort by noise so lines go left -> right in ascending order
df = df.sort_values("noise")

# plt.figure(figsize=(6, 4))
# plt.figure(figsize=(8, 6))
plt.figure(figsize=(10, 6))

# Plot each Algorithm with a distinct color & marker
sns.lineplot(
    data=df,
    x="noise",
    y="acc",
    hue="Algorithm",
    style="Algorithm",
    markers=True,   # put a marker at each data point
    # dashes=False,    # use solid lines instead of dashes
    markersize=18,
    linewidth=3
)

# Label axes, add grid
plt.xlabel("Noise Level", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
# plt.title("Accuracy vs. Noise Level (Decimal Accuracies)")
plt.grid(True, linestyle=":", linewidth=1)

# Ticks at just the noise values we haave
plt.xticks([0.1, 0.25], fontsize=20)
plt.yticks(fontsize=20)

# for alg, group in df.groupby("Algorithm"):
#     # Ensure we have both points
#     if len(group) < 2:
#         continue
#     group = group.sort_values("noise")
#     acc_low = group[group["noise"] == 0.1]["acc"].values[0]
#     acc_high = group[group["noise"] == 0.25]["acc"].values[0]
#     gradient = (acc_high - acc_low) / 0.15

#     # Place the text at the end of the line: use x slightly offset from 0.25, and y equal to the value at 0.25
#     x_text = 0.25 + 0.01  # offset to the right of 0.25
#     y_text = acc_high
#     plt.text(x_text, y_text, f"grad = {gradient:.3f}", fontsize=4,
#              color="black", ha="left", va="center",
#              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))


# # First, compute each algorithm's accuracy at noise=0.25
# annotations = []
# for alg, group in df.groupby("Algorithm"):
#     group = group.sort_values("noise")
#     # Ensure both points exist:
#     if (group["noise"] == 0.1).any() and (group["noise"] == 0.25).any():
#         acc_high = group[group["noise"] == 0.25]["acc"].values[0]
#         annotations.append((alg, acc_high))

# # Sort annotations by the y-value (accuracy at noise=0.25)
# annotations = sorted(annotations, key=lambda x: x[1])

# # Now assign offsets to prevent overlapping
# offsets = {}
# min_spacing = 0.005  # minimum vertical spacing between annotations
# current_y = None
# for alg, y_val in annotations:
#     if current_y is None:
#         new_y = y_val
#     else:
#         new_y = max(y_val, current_y + min_spacing)
#     offsets[alg] = new_y - y_val  # offset to be added to the original y_val
#     current_y = new_y

# # Now annotate each algorithm's line at the right end (x slightly > 0.25)
# for alg, group in df.groupby("Algorithm"):
#     group = group.sort_values("noise")
#     if not ((group["noise"] == 0.1).any() and (group["noise"] == 0.25).any()):
#         continue
#     acc_low = group[group["noise"] == 0.1]["acc"].values[0]
#     acc_high = group[group["noise"] == 0.25]["acc"].values[0]
#     gradient = (acc_high - acc_low) / 0.15  # denominator = 0.25 - 0.1
#     # x position: slightly to the right of 0.25
#     x_text = 0.25 + 0.005
#     # y position: acc_high plus its computed offset
#     y_text = acc_high + offsets.get(alg, 0)
#     plt.text(x_text, y_text, f"grad = {gradient:.3f}", fontsize=12,
#              color="black", ha="left", va="center",
#              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fontsize=15)

plt.tight_layout()
plt.savefig("./results_a3w/figures/noise_gradient_plot_wider.png")
