# ---- graph for looking at different distribution for different data (single plot for one class) ----
import os
import re
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

# DomainBed imports
from domainbed import algorithms, datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_unique_labels_domains(checkpoint_dir_paths):
    """ Get unique class labels and domains across all models """
    all_labels_set = set()
    all_domains_set = set()

    for checkpoint_dir_path in checkpoint_dir_paths:
        checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Collect labels
        labels = checkpoint["model_dict"].get("labels", None)
        if labels is not None:
            all_labels_set.update(labels.numpy().tolist())

        # Collect unique domains
        args = checkpoint["args"]
        data_name = args["dataset"].split("&")[0]
        dataset_class = vars(datasets)[data_name]
        dataset_obj = dataset_class(args["data_dir"], args["test_envs"], checkpoint["model_hparams"])

        for env_i, _ in enumerate(dataset_obj):
            all_domains_set.add(env_i)

    return sorted(all_labels_set), sorted(all_domains_set)

def graph_tsne_single_target(checkpoint_dir_path, label_to_color, domain_to_marker):
    checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    folder_name = os.path.basename(checkpoint_dir_path)

    input_shape   = checkpoint["model_input_shape"]
    num_classes   = checkpoint["model_num_classes"]
    num_domains   = checkpoint["model_num_domains"]
    model_hparams = checkpoint["model_hparams"]
    args          = checkpoint["args"]
    
    # Remove noise settings
    if model_hparams.get('flip_prob') is not None:
        model_hparams['flip_prob'] = 0
        model_hparams['study_noise'] = 0
    print(f"Model hparams after removing noise: {model_hparams}")

    state_dict = {k: v for k, v in checkpoint["model_dict"].items() if "maplayers" not in k}

    model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    featurizer = model.featurizer.to(device)

    data_name = args["dataset"].split("&")[0]
    dataset_class = vars(datasets)[data_name]
    dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)

    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
        if free_mem_gb > 20:
            print("Detected large GPU memory. Setting dataset_obj.N_WORKERS = 1")
            dataset_obj.N_WORKERS = 0

    in_splits, out_splits = [], []
    for env_i, env in enumerate(dataset_obj):
        out, in_ = misc.split_dataset(env, int(len(env) * args['holdout_fraction']),
                                        misc.seed_hash(args['trial_seed'], env_i))
        if env_i in args['test_envs']:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args['uda_holdout_fraction']),
                                            misc.seed_hash(args['trial_seed'], env_i))
        in_splits.append((in_, None))
        if len(out) > 0:
            out_splits.append((out, None))

    test_in_splits = [in_splits[i] for i in range(len(in_splits)) if i in args["test_envs"]]
    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
        for env, _ in (test_in_splits + out_splits)
    ]

    eval_loader_names = [f"env{i}_in" for i in range(len(in_splits)) if i in args["test_envs"]]
    eval_loader_names += [f"env{i}_out" for i in range(len(out_splits))]
    evals = zip(eval_loader_names, eval_loaders)

    all_features, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for name, loader in evals:
            match = re.search(r"env(\d+)_", name)
            domain_i = int(match.group(1)) if match else -1
            for x, y in loader:
                x = x.to(device)
                feats = featurizer(x).cpu()
                all_features.append(feats)
                all_labels.append(y.cpu())
                all_domains.append(torch.full_like(y, domain_i))

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels   = torch.cat(all_labels, dim=0).numpy()
    all_domains  = torch.cat(all_domains, dim=0).numpy()

    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(all_features)

    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label": all_labels.astype(str),
        "domain": all_domains.astype(str)
    })

    # Choose a target class label to inspect (e.g., the first one).
    unique_labels = sorted(df["label"].unique())
    target_label = unique_labels[0]

    # Filter DataFrame for the target class.
    sub_df = df[df["label"] == target_label]

    # Plot all domains for this target class on one graph.
    plt.figure(figsize=(10,8))
    sns.scatterplot(
        data=sub_df,
        x="x",
        y="y",
        hue="domain",       # Domain determines the color.
        palette="tab10",
        s=120,
        alpha=0.8,
        edgecolor="k",
    )
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.axis('off')
    plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"results/figures_comparison/domainComp_{args['algorithm']}_{data_name}_tsne_singleTarget_larger.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir_paths = [
        "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
        # "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
        # "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
    ]

    # Step 1: Extract unique labels & domains across all models
    unique_labels, unique_domains = get_unique_labels_domains(checkpoint_dir_paths)

    # Step 2: Create fixed color & marker mappings (for use in other plots, if needed)
    fixed_palette = sns.color_palette("tab10", len(unique_labels))
    label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}

    fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]
    domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(unique_domains)}

    # Step 3: Plot the t-SNE for a target class across all domains on a single graph.
    for path in checkpoint_dir_paths:
        graph_tsne_single_target(path, label_to_color, domain_to_marker)


# import copy
# import os
# import re
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import pandas as pd
# from sklearn.manifold import TSNE

# # DomainBed imports
# from domainbed import algorithms, datasets
# from domainbed.lib.fast_data_loader import FastDataLoader
# from domainbed.lib import misc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def get_unique_labels_domains(checkpoint_dir_paths):
#     """ Get unique class labels and domains across all models """
#     all_labels_set = set()
#     all_domains_set = set()
#     for checkpoint_dir_path in checkpoint_dir_paths:
#         checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#         checkpoint = torch.load(checkpoint_path, map_location="cpu")
#         # Collect labels
#         labels = checkpoint["model_dict"].get("labels", None)
#         if labels is not None:
#             all_labels_set.update(labels.numpy().tolist())
#         # Collect unique domains
#         args = checkpoint["args"]
#         data_name = args["dataset"].split("&")[0]
#         dataset_class = vars(datasets)[data_name]
#         dataset_obj = dataset_class(args["data_dir"], args["test_envs"], checkpoint["model_hparams"])
#         for env_i, _ in enumerate(dataset_obj):
#             all_domains_set.add(env_i)
#     return sorted(all_labels_set), sorted(all_domains_set)

# def graph_tsne_single_target_noise(checkpoint_dir_path, label_to_color):
#     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     folder_name = os.path.basename(checkpoint_dir_path)
    
#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     orig_model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]
    
#     # Create two sets of hyperparameters:
#     # 1. Noise preserved (original)
#     hparams_noise = orig_model_hparams
#     # 2. Noise removed: set flip_prob and study_noise to 0
#     hparams_no_noise = orig_model_hparams.copy()
#     if hparams_no_noise.get('flip_prob') is not None:
#         hparams_no_noise['flip_prob'] = 0
#         hparams_no_noise['study_noise'] = 0
#     print("Using two conditions: noise preserved vs. noise removed.")
    
#     # Load state_dict (exclude mapping layers)
#     state_dict = {k: v for k, v in checkpoint["model_dict"].items() if "maplayers" not in k}
    
#     # Instantiate two models
#     model_noise = algorithms.NLPGERM(input_shape, num_classes, num_domains, hparams_noise)
#     model_no_noise = algorithms.NLPGERM(input_shape, num_classes, num_domains, hparams_no_noise)
#     model_noise.load_state_dict(state_dict, strict=False)
#     model_no_noise.load_state_dict(state_dict, strict=False)
#     model_noise.eval()
#     model_no_noise.eval()
    
#     featurizer_noise = model_noise.featurizer.to(device)
#     featurizer_no_noise = model_no_noise.featurizer.to(device)
    
#     # Load dataset
#     data_name = args["dataset"].split("&")[0]
#     dataset_class = vars(datasets)[data_name]
#     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], orig_model_hparams)
    
#     # if torch.cuda.is_available():
#     #     free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#     #     if free_mem_gb > 20:
#     #         print("Detected large GPU memory. Setting dataset_obj.N_WORKERS = 1")
#     #         dataset_obj.N_WORKERS = 0
    
#     in_splits, out_splits = [], []
#     for env_i, env in enumerate(dataset_obj):
#         out, in_ = misc.split_dataset(env, int(len(env) * args['holdout_fraction']),
#                                         misc.seed_hash(args['trial_seed'], env_i))
#         if env_i in args['test_envs']:
#             uda, in_ = misc.split_dataset(in_, int(len(in_) * args['uda_holdout_fraction']),
#                                             misc.seed_hash(args['trial_seed'], env_i))
#         in_splits.append((in_, None))
#         if len(out) > 0:
#             out_splits.append((out, None))
    
#     test_in_splits = [in_splits[i] for i in range(len(in_splits)) if i in args["test_envs"]]
#     eval_loaders = [
#         FastDataLoader(dataset=env, batch_size=hparams_noise["test_batch_size"],
#                        num_workers=dataset_obj.N_WORKERS)
#         for env, _ in (test_in_splits + out_splits)
#     ]
    
#     eval_loader_names = [f"env{i}_in" for i in range(len(in_splits)) if i in args["test_envs"]]
#     eval_loader_names += [f"env{i}_out" for i in range(len(out_splits))]
#     evals = zip(eval_loader_names, eval_loaders)
    
#     features_list = []
#     labels_list = []
#     domains_list = []
#     noise_condition = []  # "preserved" or "removed"
    
#     with torch.no_grad():
#         for name, loader in evals:
#             match = re.search(r"env(\d+)_", name)
#             domain_i = int(match.group(1)) if match else -1
#             for x, y in loader:
#                 x = x.to(device)
#                 # Extract features with noise preserved.
#                 feats_noise = featurizer_noise(x).cpu()
#                 # Extract features with noise removed.
#                 feats_no_noise = featurizer_no_noise(x).cpu()
#                 # Append both sets (each sample is added twice, with its noise condition)
#                 features_list.append(feats_noise)
#                 labels_list.append(y.cpu())
#                 domains_list.append(torch.full_like(y, domain_i))
#                 noise_condition.extend(["preserved"] * y.size(0))
                
#                 features_list.append(feats_no_noise)
#                 labels_list.append(y.cpu())
#                 domains_list.append(torch.full_like(y, domain_i))
#                 noise_condition.extend(["removed"] * y.size(0))
                
#     print(noise_condition)
    
#     all_features = torch.cat(features_list, dim=0).numpy()
#     all_labels   = torch.cat(labels_list, dim=0).numpy()
#     all_domains  = torch.cat(domains_list, dim=0).numpy()
    
#     tsne = TSNE(n_components=2, random_state=0)
#     features_2d = tsne.fit_transform(all_features)
    
#     df = pd.DataFrame({
#         "x": features_2d[:, 0],
#         "y": features_2d[:, 1],
#         "label": all_labels.astype(str),
#         "domain": all_domains.astype(str),
#         "noise": noise_condition
#     })
    
#     # Choose a target class label to inspect (e.g., the first one).
#     unique_labels = sorted(df["label"].unique())
#     target_label = unique_labels[0]
#     sub_df = df[df["label"] == target_label]
#     print(sub_df["noise"])
    
#     sub_df = sub_df.sample(frac=1, random_state=0)
    
#     plt.figure(figsize=(10,8))
#     sns.scatterplot(
#         data=sub_df,
#         x="x",
#         y="y",
#         hue="domain",          # Domain determines color.
#         style="noise",         # Noise condition determines marker shape.
#         style_order=["preserved", "removed"],
#         markers={"preserved": "o", "removed": "s"},
#         palette="tab10",
#         s=60,
#         alpha=0.8,
#         edgecolor="k"
#     )
#     plt.title("")
#     plt.xlabel("")
#     plt.ylabel("")
#     plt.legend(title="Domain / Noise", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(f"results/figures_comparison/domainComp_{args['algorithm']}_{data_name}_tsne_singleTarget_noise.png",
#                 bbox_inches="tight", dpi=300)
#     plt.show()
    
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir_paths = [
#         "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         # Additional checkpoint paths can be added here.
#     ]
    
#     # Create a color mapping based on unique domains.
#     unique_labels, unique_domains = get_unique_labels_domains(checkpoint_dir_paths)
#     fixed_palette = sns.color_palette("tab10", len(unique_domains))
#     label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}
    
#     for path in checkpoint_dir_paths:
#         graph_tsne_single_target_noise(path, label_to_color)
