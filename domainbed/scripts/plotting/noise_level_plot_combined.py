# Plot differnet noise level vs acc
#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re

# DomainBed imports
from domainbed import model_selection
from domainbed.lib import misc, reporting

def get_test_trajectory_for_dir(input_dir, selection_method):
    """
    Loads DomainBed records from `input_dir`, applies `selection_method`,
    and returns a single 1D NumPy array representing test accuracy vs. epoch.
    This example picks the first group (and if multiple groups, the first one).
    """
    records = reporting.load_records(input_dir)
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    if 'hparams tracker' in selection_method.name.lower():
        grouped_records = reporting.get_grouped_records(records).map(lambda group:
            {
                **group,
                "val_trajectories": selection_method.sweep_trajectory(group["records"])[0],
                "test_trajectories": selection_method.sweep_trajectory(group["records"])[1],
                "train_trajectories": selection_method.sweep_trajectory(group["records"])[2]
            }
        ).filter(lambda g: g["test_trajectories"] is not None)

    if not len(grouped_records):
        print(f"No valid grouped records in {input_dir}")
        return None

    first_group = grouped_records[0]
    test_trajectories = first_group.get("test_trajectories", None)
    if test_trajectories is None:
        print(f"No test_trajectories found in {input_dir}")
        return None

    test_trajectories = np.array(test_trajectories)
    if test_trajectories.ndim == 2:
        # If there are multiple groups, select the first one.
        single_line = test_trajectories[0]
    else:
        single_line = test_trajectories

    return single_line

def extract_noise_value(path):
    """
    Given a directory path like '.../noise0.3', extract the numeric portion (0.3).
    Returns a float so we can sort by ascending noise level.
    """
    base = os.path.basename(path)  # e.g. "noise0.3"
    match = re.search(r"noise([\d\.]+)", base)
    if match:
        return float(match.group(1))  # Convert "0.3" -> 0.3
    return 9999.0  # Default if no match

def main():
    parser = argparse.ArgumentParser(
        description="Plot test accuracy vs. epoch for each subdirectory of an input directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Top-level directory containing subdirectories with DomainBed results.")
    parser.add_argument("--output", type=str, default="combined_plot.png",
                        help="Filename for the output PNG image.")
    parser.add_argument("--tracker", type=str, default="OracleHparamTracker",
                        help="Selection method to use (e.g., OracleHparamTracker, ValWGHparamTracker).")
    args = parser.parse_args()

    # Map tracker string to selection method class.
    method_map = {
        "OracleHparamTracker": model_selection.OracleHparamTracker,
        "ValWGHparamTracker": model_selection.ValWGHparamTracker,
    }
    selection_cls = method_map.get(args.tracker, model_selection.OracleHparamTracker)

    # List subdirectories
    all_subdirs = [
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ]

    # Sort by numeric noise value
    all_subdirs = sorted(all_subdirs, key=extract_noise_value)
    
    markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10", len(all_subdirs))

    for i, subdir in enumerate(all_subdirs):
        noise_val = extract_noise_value(subdir)
        trajectory = get_test_trajectory_for_dir(subdir, selection_cls)
        if trajectory is None:
            continue
        epochs = np.arange(len(trajectory))
        # Label now includes "noise level = <value>"
        label = f"nl = {noise_val}"
        plt.plot(epochs, trajectory, label=label, color=colors(i), marker=markers[i])
        
    # 1) After collecting/plotting all trajectories, determine max_epochs
    max_epochs = 0
    for i, subdir in enumerate(all_subdirs):
        trajectory = get_test_trajectory_for_dir(subdir, selection_cls)
        if trajectory is not None:
            max_epochs = max(max_epochs, len(trajectory))

    # # 2) Once you know the largest number of epochs,
    # #    set x-axis ticks every 300 epochs.
    # tick_step = 300
    # plt.xticks(np.arange(0, max_epochs + 1, tick_step))
    print(f"max epoch: {max_epochs}")
    num_ticks = 10
    tick_step = max(1, max_epochs // num_ticks)
    print(tick_step)

    plt.xticks(np.arange(0, max_epochs + 1, tick_step), fontsize=15)
    # plt.ylim((0.4, 0.8))
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle=':', linewidth=1, color='gray')

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Test Accuracy", fontsize=24)
    # plt.title("Test Accuracy vs. Epoch (Sorted by Noise)")
    # plt.legend(loc="upper right")
    plt.legend(
        loc="center left",      # anchor the legend's left side
        bbox_to_anchor=(1.0, 0.5),  # x=1.0 is the right edge of the axes, y=0.5 is the vertical center
        fontsize=15
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()
    
if __name__ == "__main__":
    main()
    
    # command: python -m domainbed.scripts.plotting.plot_results_noise_combined --input_dir ./results/nlpgerm_oh_noise_level_analysis_v2/ --tracker OracleHparamTracker --output results/figures/noise_level_combined_plot_larger.png