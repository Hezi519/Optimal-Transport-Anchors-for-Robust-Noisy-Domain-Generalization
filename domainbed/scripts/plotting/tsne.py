# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '9'

# import re
# import torch
# import torch.nn as nn
# import torchvision
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.manifold import TSNE
# import seaborn as sns
# import pandas as pd

# # DomainBed imports
# from domainbed import algorithms, datasets
# from domainbed.lib.fast_data_loader import FastDataLoader
# from domainbed.lib import misc

# def graph_tsne(checkpoint_dir_path):
#     checkpoint_path = checkpoint_dir_path + "/model_best.pkl"
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     folder_name = checkpoint_path.split('/')[-2]

#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]

#     # Remove any "maplayers" entries if needed
#     state_dict = {
#         k: v for k, v in checkpoint["model_dict"].items()
#         if "maplayers" not in k
#     }

#     model = algorithms.NLPGERM(
#         input_shape, num_classes, num_domains, model_hparams
#     )
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()

#     featurizer = model.featurizer.to(device)

#     data_name = args["dataset"].split("&")[0]
#     dataset_class = vars(datasets)[data_name]
#     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)

#     if torch.cuda.is_available():
#         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#         if free_mem_gb > 20:
#             print("Detected large GPU memory. Setting dataset_obj.N_WORKERS = 1")
#             dataset_obj.N_WORKERS = 0

#     in_splits = []
#     out_splits = []
#     for env_i, env in enumerate(dataset_obj):
#         out, in_ = misc.split_dataset(
#             env,
#             int(len(env) * args['holdout_fraction']),
#             misc.seed_hash(args['trial_seed'], env_i)
#         )
#         if env_i in args['test_envs']:
#             uda, in_ = misc.split_dataset(
#                 in_,
#                 int(len(in_) * args['uda_holdout_fraction']),
#                 misc.seed_hash(args['trial_seed'], env_i)
#             )
#         in_weights, out_weights, uda_weights = None, None, None
#         in_splits.append((in_, in_weights))
#         if len(out) > 0:
#             out_splits.append((out, out_weights))

#     test_in_splits = [
#         in_splits[i] for i in range(len(in_splits))
#         if i in args["test_envs"]
#     ]
#     eval_loaders = [
#         FastDataLoader(
#             dataset=env,
#             batch_size=model_hparams["test_batch_size"],
#             num_workers=dataset_obj.N_WORKERS
#         )
#         for env, _ in (test_in_splits + out_splits)
#     ]

#     eval_loader_names = [
#         f"env{i}_in" for i in range(len(in_splits)) if i in args["test_envs"]
#     ]
#     eval_loader_names += [
#         f"env{i}_out" for i in range(len(out_splits))
#     ]
#     eval_weights = [None for _ in (test_in_splits + out_splits)]
#     evals = zip(eval_loader_names, eval_loaders, eval_weights)

#     all_features = []
#     all_labels   = []
#     all_domains  = []

#     with torch.no_grad():
#         for name, loader, _ in evals:
#             match = re.search(r"env(\d+)_", name)
#             domain_i = int(match.group(1)) if match else -1

#             for x, y in loader:
#                 x = x.to(device)
#                 feats = featurizer(x).cpu()
#                 all_features.append(feats)
#                 all_labels.append(y.cpu())

#                 # domain array for this batch
#                 domain_arr = torch.full_like(y, domain_i)
#                 all_domains.append(domain_arr)

#     all_features = torch.cat(all_features, dim=0)  # shape [N, feat_dim]
#     all_labels   = torch.cat(all_labels, dim=0)    # shape [N]
#     all_domains  = torch.cat(all_domains, dim=0)   # shape [N]

#     # Convert to NumPy for t-SNE
#     features_np = all_features.numpy()
#     labels_np   = all_labels.numpy()
#     domains_np  = all_domains.numpy()

#     tsne = TSNE(n_components=2, random_state=0)
#     features_2d = tsne.fit_transform(features_np)

#     # Build a DataFrame
#     df = pd.DataFrame({
#         "x": features_2d[:, 0],
#         "y": features_2d[:, 1],
#         "label": labels_np.astype(str),
#         "domain": domains_np.astype(str)
#     })

#     plt.figure(figsize=(12, 10))
#     sns.scatterplot(
#         data=df,
#         x="x",
#         y="y",
#         hue="label",      # color by class
#         style="domain",   # shape by domain
#         palette="tab10",  # up to 20 distinct colors
#         alpha=0.7
#     )
#     plt.title("t-SNE: Color=label, Shape=domain")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(f"results/figures_comparison/{args['algorithm']}_{data_name}_tsne_{folder_name}.png", bbox_inches="tight", dpi=300)
#     # plt.show()




# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir_paths = ["./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#         "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"]
#     for path in checkpoint_dir_paths:
#         graph_tsne(path)
        
        
#     # checkpoint_dir_paths = ["./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#     #     "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#     #     "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044",
#     #     "./results/nlpgerm_vlcs_ma/382ca01561aee1c10c464f7d81f78a25",
#     #     "./results/erm_vlcs_baseline/b1d74fe21a7a8a3a275ace372339cbce",
#     #     "./results/mixup_vlcs_baseline/b84182d3ea9d086e1bf1442f76f2126e",
#     #     "./results/nlpgerm_oh_ma/84271b299684bca42a731497dc7a5b27",
#     #     "./results/erm_oh_baseline/79982d75de9589f0116c45c209af954c",
#     #     "./results/mixup_oh_baseline/4d119284ba425e9736065b180eb83347",
#     #     "./results/nlpgerm_terra_ma/be6bbb69e9c6851dc25a3dd5000a38b4",
#     #     "./results/erm_terra_baseline/707d05536e59887ec9a6e1c19840dcc5",
#     #     "./results/mixup_terra_baseline/fa25c3d5fe31246ca50a8f3231be7059"]
    
# ---- graph without train/test split ---- 
# import os
# import re
# import torch
# import torch.nn as nn
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

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


# def graph_tsne(checkpoint_dir_path, label_to_color, domain_to_marker):
#     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     folder_name = os.path.basename(checkpoint_dir_path)

#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]

#     state_dict = {k: v for k, v in checkpoint["model_dict"].items() if "maplayers" not in k}

#     model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     featurizer = model.featurizer.to(device)

#     data_name = args["dataset"].split("&")[0]
#     dataset_class = vars(datasets)[data_name]
#     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)

#     if torch.cuda.is_available():
#         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#         if free_mem_gb > 20:
#             print("Detected large GPU memory. Setting dataset_obj.N_WORKERS = 1")
#             dataset_obj.N_WORKERS = 0

#     in_splits, out_splits = [], []
#     for env_i, env in enumerate(dataset_obj):
#         out, in_ = misc.split_dataset(env, int(len(env) * args['holdout_fraction']), misc.seed_hash(args['trial_seed'], env_i))
#         if env_i in args['test_envs']:
#             uda, in_ = misc.split_dataset(in_, int(len(in_) * args['uda_holdout_fraction']), misc.seed_hash(args['trial_seed'], env_i))
#         in_splits.append((in_, None))
#         if len(out) > 0:
#             out_splits.append((out, None))

#     test_in_splits = [in_splits[i] for i in range(len(in_splits)) if i in args["test_envs"]]
#     eval_loaders = [
#         FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
#         for env, _ in (test_in_splits + out_splits)
#     ]

#     eval_loader_names = [f"env{i}_in" for i in range(len(in_splits)) if i in args["test_envs"]]
#     eval_loader_names += [f"env{i}_out" for i in range(len(out_splits))]
#     evals = zip(eval_loader_names, eval_loaders)

#     all_features, all_labels, all_domains = [], [], []
#     with torch.no_grad():
#         for name, loader in evals:
#             match = re.search(r"env(\d+)_", name)
#             domain_i = int(match.group(1)) if match else -1

#             for x, y in loader:
#                 x = x.to(device)
#                 feats = featurizer(x).cpu()
#                 all_features.append(feats)
#                 all_labels.append(y.cpu())
#                 all_domains.append(torch.full_like(y, domain_i))

#     all_features = torch.cat(all_features, dim=0).numpy()
#     all_labels   = torch.cat(all_labels, dim=0).numpy()
#     all_domains  = torch.cat(all_domains, dim=0).numpy()

#     tsne = TSNE(n_components=2, random_state=0)
#     features_2d = tsne.fit_transform(all_features)

#     df = pd.DataFrame({
#         "x": features_2d[:, 0],
#         "y": features_2d[:, 1],
#         "label": all_labels.astype(str),
#         "domain": all_domains.astype(str)
#     })

#     # Ensure all labels are covered
#     all_labels_in_data = sorted(df["label"].unique())
#     all_domains_in_data = sorted(df["domain"].unique())

#     # Update label-to-color mapping dynamically
#     fixed_palette = sns.color_palette("tab10", len(all_labels_in_data))
#     label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(all_labels_in_data)}

#     # Ensure all domains have a shape assigned
#     fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]
#     domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(all_domains_in_data)}

#     plt.figure(figsize=(12, 10))
#     sns.scatterplot(
#         data=df,
#         x="x",
#         y="y",
#         hue="label",
#         style="domain",
#         palette=label_to_color,
#         markers=domain_to_marker,
#         alpha=0.7
#     )
#     plt.title(f"t-SNE: {args['algorithm']} - Color=label, Shape=domain")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(f"results/figures_comparison/v2_{args['algorithm']}_{data_name}_tsne_{folder_name}.png", bbox_inches="tight", dpi=300)
#     plt.show()


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir_paths = [
#         "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#         "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
#     ]

#     # Step 1: Extract unique labels & domains across all models
#     unique_labels, unique_domains = get_unique_labels_domains(checkpoint_dir_paths)

#     # Step 2: Create fixed color & marker mappings
#     fixed_palette = sns.color_palette("tab10", len(unique_labels))  # Fixed color mapping
#     label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}

#     fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]  # Fixed shape mapping
#     domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(unique_domains)}

#     # Step 3: Plot each model with fixed mappings
#     for path in checkpoint_dir_paths:
#         graph_tsne(path, label_to_color, domain_to_marker)


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

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

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


def graph_tsne(checkpoint_dir_path, label_to_color, domain_to_marker, dataset_type):
    """ Generate a t-SNE plot for either train or test dataset of a given model """
    checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    folder_name = os.path.basename(checkpoint_dir_path)

    input_shape   = checkpoint["model_input_shape"]
    num_classes   = checkpoint["model_num_classes"]
    num_domains   = checkpoint["model_num_domains"]
    model_hparams = checkpoint["model_hparams"]
    args          = checkpoint["args"]

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
        out, in_ = misc.split_dataset(env, int(len(env) * args['holdout_fraction']), misc.seed_hash(args['trial_seed'], env_i))
        if env_i in args['test_envs']:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args['uda_holdout_fraction']), misc.seed_hash(args['trial_seed'], env_i))
        in_splits.append((in_, None))
        if len(out) > 0:
            out_splits.append((out, None))

    if dataset_type == "train":
        selected_splits = in_splits  # Train dataset
        eval_loader_names = [f"env{i}_train" for i in range(len(in_splits))]
    else:
        selected_splits = out_splits  # Test dataset
        eval_loader_names = [f"env{i}_test" for i in range(len(out_splits))]

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
        for env, _ in selected_splits
    ]

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

    if len(all_features) == 0:
        print(f"Skipping {dataset_type} set for {folder_name} (no data available)")
        return

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

    all_labels_in_data = sorted(df["label"].unique())
    all_domains_in_data = sorted(df["domain"].unique())

    # Update label-to-color mapping dynamically
    fixed_palette = sns.color_palette("tab10", len(all_labels_in_data))
    label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(all_labels_in_data)}

    # Ensure all domains have a shape assigned
    fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]
    domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(all_domains_in_data)}

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        style="domain",
        palette=label_to_color,
        markers=domain_to_marker,
        alpha=0.7
    )
    plt.title(f"t-SNE: {args['algorithm']} ({dataset_type} set) - Color=label, Shape=domain")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"results_a3w/figures_comparison/v3_{dataset_type}_{args['algorithm']}_{data_name}_tsne_{folder_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir_paths = [
        # # "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
        # # "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
        # # "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
        # "./results/nlpgerm_vlcs_ma/382ca01561aee1c10c464f7d81f78a25",
        # "./results/erm_vlcs_baseline/b1d74fe21a7a8a3a275ace372339cbce",
        # "./results/mixup_vlcs_baseline/b84182d3ea9d086e1bf1442f76f2126e",
        "./result_a3w/irm_vlcs/0eff050fbf9ae2b44ae55a40dccf028c"
    ]

    # Step 1: Extract unique labels & domains across all models
    unique_labels, unique_domains = get_unique_labels_domains(checkpoint_dir_paths)

    # Step 2: Create fixed color & marker mappings
    fixed_palette = sns.color_palette("tab10", len(unique_labels))  # Fixed color mapping
    label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}

    fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]  # Fixed shape mapping
    domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(unique_domains)}

    # Step 3: Plot each model for both train and test sets
    for path in checkpoint_dir_paths:
        graph_tsne(path, label_to_color, domain_to_marker, dataset_type="train")
        graph_tsne(path, label_to_color, domain_to_marker, dataset_type="test")

