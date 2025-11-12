import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# lambda
# data = [
#     {"Algorithm": "Ours",       "lambda": 0.1,  "acc": 0.834},
#     {"Algorithm": "Ours",       "lambda": 0.5, "acc": 0.804},
#     {"Algorithm": "Ours",       "lambda": 1.0, "acc": 0.807},
#     {"Algorithm": "ERM",     "lambda": 0.1,  "acc": 0.681},
#     {"Algorithm": "ERM",     "lambda": 0.5, "acc": 0.701},
#     {"Algorithm": "ERM",     "lambda": 1.0, "acc": 0.700},
#     {"Algorithm": "Mixup",      "lambda": 0.1,  "acc": 0.622},
#     {"Algorithm": "Mixup",      "lambda": 0.5, "acc": 0.688},
#     {"Algorithm": "Mixup",      "lambda": 1.0, "acc": 0.637},
# ]

# # lr
# data = [
#     {"Algorithm": "Ours",       "lr":1e-2,  "acc": 0.147},
#     {"Algorithm": "Ours",       "lr":1e-3, "acc": 0.120},
#     {"Algorithm": "Ours",       "lr":1e-4, "acc": 0.786},
#     {"Algorithm": "Ours",       "lr":1e-5,  "acc": 0.718},
#     {"Algorithm": "Ours",       "lr":1e-6, "acc": 0.33},
#     {"Algorithm": "ERM",       "lr":1e-2,  "acc": 0.132},
#     {"Algorithm": "ERM",       "lr":1e-3, "acc": 0.223},
#     {"Algorithm": "ERM",       "lr":1e-4, "acc": 0.614},
#     {"Algorithm": "ERM",       "lr":1e-5,  "acc": 0.694},
#     {"Algorithm": "ERM",       "lr":1e-6, "acc": 0.684},
#     {"Algorithm": "Mixup",       "lr":1e-2,  "acc": 0.127},
#     {"Algorithm": "Mixup",       "lr":1e-3, "acc": 0.099},
#     {"Algorithm": "Mixup",       "lr":1e-4, "acc": 0.608},
#     {"Algorithm": "Mixup",       "lr":1e-5,  "acc": 0.648},
#     {"Algorithm": "Mixup",       "lr":1e-6, "acc": 0.739},
# ]

# # noise
# data = [
#     {"Algorithm": "AAAW",       "noise":0,  "acc": 0.75},
#     {"Algorithm": "AAAW",       "noise":0.1, "acc": 0.763},
#     {"Algorithm": "AAAW",       "noise":0.2, "acc": 0.713},
#     {"Algorithm": "AAAW",       "noise":0.3,  "acc": 0.608},
#     {"Algorithm": "AAAW",       "noise":0.4, "acc": 0.627},
#     {"Algorithm": "AAAW",       "noise":0.5, "acc": 0.517},
#     {"Algorithm": "ERM",       "noise":0,  "acc": 0.782},
#     {"Algorithm": "ERM",       "noise":0.1, "acc": 0.745},
#     {"Algorithm": "ERM",       "noise":0.2, "acc": 0.688},
#     {"Algorithm": "ERM",       "noise":0.3,  "acc": 0.612},
#     {"Algorithm": "ERM",       "noise":0.4, "acc": 0.558},
#     {"Algorithm": "ERM",       "noise":0.5, "acc": 0.351}
#     {"Algorithm": "AAAW",       "noise":0,  "acc": 0.836},
#     {"Algorithm": "AAAW",       "noise":0.1, "acc": 0.801},
#     {"Algorithm": "AAAW",       "noise":0.2, "acc": 0.735},
#     {"Algorithm": "AAAW",       "noise":0.3,  "acc": 0.601},
#     {"Algorithm": "AAAW",       "noise":0.4, "acc": 0.627},
# ]

# # iter_freq
# data = [
#     {"Algorithm": "Ours",       "iter_freq":5, "acc": 0.797},
#     {"Algorithm": "Ours",       "iter_freq":10, "acc": 0.807},
#     {"Algorithm": "Ours",       "iter_freq":15,  "acc": 0.778},
#     {"Algorithm": "Ours",       "iter_freq":20,  "acc": 0.815},
# ]

# temp
data = [
    {"Algorithm": "Ours",       "temp":0, "acc": 0.774},
    {"Algorithm": "Ours",       "temp":5, "acc": 0.768},
    {"Algorithm": "Ours",       "temp":10,  "acc": 0.821},
    {"Algorithm": "Ours",       "temp":15,  "acc": 0.527},
]

df = pd.DataFrame(data)
# Sort by noise so lines go left -> right in ascending order
df = df.sort_values("temp")

plt.figure(figsize=(6, 4))

# Plot each Algorithm with a distinct color & marker
sns.lineplot(
    data=df,
    x="temp",
    y="acc",
    hue="Algorithm",
    style="Algorithm",
    markers=True,   # put a marker at each data point
    # dashes=False,    # use solid lines instead of dashes
    markersize=12,
    linewidth=2
)

# Label axes, add grid
plt.xlabel("temp", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
# plt.title("Accuracy vs. Noise Level (Decimal Accuracies)")
plt.grid(True, linestyle=":", linewidth=1)

# Ticks at just the noise values we haave
plt.xticks([0, 5, 10, 15], fontsize=18)
plt.ylim((0, 1))
# import numpy as np
# xticks =  np.logspace(-6, -2, num=5)
# plt.xticks(xticks, fontsize=18)
# plt.yticks(fontsize=18)
# plt.xscale('log')

# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

plt.tight_layout()
plt.savefig("./results/figures/temp_gradient_plot_v3.png")
