import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# # Set global font to Times New Roman
# plt.rcParams["font.family"] = "Times New Roman"

# Read the file and find the header line
filename = 'out.txt'
with open(filename, 'r') as f:
    lines = f.readlines()

header_index = None
for i, line in enumerate(lines):
    if line.startswith("env0_in_acc"):
        header_index = i
        break

df = pd.read_csv(filename, delim_whitespace=True, skiprows=header_index)

# -- Identify the columns for noisy accuracy --
noisy_cols = [col for col in df.columns if re.match(r'env\d+_noisy_a', col)]

# -- Plot each environment's noisy accuracy using the "step" column for x --
plt.figure(figsize=(8, 6))
# for col in noisy_cols:
#     plt.plot(df["step"], df[col], label=col)
    
for col in noisy_cols:
    # Extract the environment number using regex.
    match = re.match(r'env(\d+)_noisy_a', col)
    if match:
        env_num = match.group(1)
        label = f"env {env_num}"
    else:
        label = col
    plt.plot(df["step"], df[col], label=label)

# -- Format the axes --
ax = plt.gca()

# Format the y-axis: show fewer ticks and limit to 2 decimals
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Format the x-axis: show ticks at multiples of 300, display them as integers
# ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
ax.xaxis.set_major_locator(ticker.MultipleLocator(600))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
ax.tick_params(axis='x', labelsize=8)
plt.xlim(0, 4800)

ax.tick_params(axis='both', labelsize=12)
plt.grid(True, linestyle=":", linewidth=1)

plt.xlabel("Step", fontsize=14)
plt.ylabel("Noisy Accuracy", fontsize=14)
plt.title("Noisy Accuracy Over Time", fontsize=16)
plt.legend(fontsize=12)
plt.savefig("./results/figures/noise_acc_grid.png")