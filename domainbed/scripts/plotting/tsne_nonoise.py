# # ---- graph without train/test split ---- 
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
    
#     # change noise to 0
#     if model_hparams['flip_prob'] != None:
#         model_hparams['flip_prob'] = 0
#         model_hparams['study_noise'] = 0
#     print(f"Model hparams after removing noise: {model_hparams}")

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
#     plt.savefig(f"results/figures_comparison/v4_{args['algorithm']}_{data_name}_tsne_{folder_name}_nonoise.png", bbox_inches="tight", dpi=300)
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

# ---- graph with train/test split ----
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


# def graph_tsne(checkpoint_dir_path, label_to_color, domain_to_marker, dataset_type):
#     """ Generate a t-SNE plot for either train or test dataset of a given model """
#     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     folder_name = os.path.basename(checkpoint_dir_path)

#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]
    
#     # change noise to 0
#     if model_hparams['flip_prob'] != None:
#         model_hparams['flip_prob'] = 0
#         model_hparams['study_noise'] = 0
#     print(f"Model hparams after removing noise: {model_hparams}")

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

#     if dataset_type == "train":
#         selected_splits = in_splits  # Train dataset
#         eval_loader_names = [f"env{i}_train" for i in range(len(in_splits))]
#     else:
#         selected_splits = out_splits  # Test dataset
#         eval_loader_names = [f"env{i}_test" for i in range(len(out_splits))]

#     eval_loaders = [
#         FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
#         for env, _ in selected_splits
#     ]

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

#     if len(all_features) == 0:
#         print(f"Skipping {dataset_type} set for {folder_name} (no data available)")
#         return

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
#     # plt.title(f"t-SNE: {args['algorithm']} ({dataset_type} set) - Color=label, Shape=domain")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="center left")
#     plt.tight_layout()
#     plt.savefig(f"results/figures_comparison/v6_{dataset_type}_{args['algorithm']}_{data_name}_tsne_{folder_name}_nonoise.png", bbox_inches="tight", dpi=300)
#     plt.show()


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir_paths = [
#         "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#         "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
#         # "./results/nlpgerm_vlcs_ma/382ca01561aee1c10c464f7d81f78a25",
#         # "./results/erm_vlcs_baseline/b1d74fe21a7a8a3a275ace372339cbce",
#         # "./results/mixup_vlcs_baseline/b84182d3ea9d086e1bf1442f76f2126e",
#     ]

#     # Step 1: Extract unique labels & domains across all models
#     unique_labels, unique_domains = get_unique_labels_domains(checkpoint_dir_paths)

#     # Step 2: Create fixed color & marker mappings
#     fixed_palette = sns.color_palette("tab10", len(unique_labels))  # Fixed color mapping
#     label_to_color = {str(label): fixed_palette[i] for i, label in enumerate(unique_labels)}

#     fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]  # Fixed shape mapping
#     domain_to_marker = {str(domain): fixed_markers[i % len(fixed_markers)] for i, domain in enumerate(unique_domains)}

#     # Step 3: Plot each model for both train and test sets
#     for path in checkpoint_dir_paths:
#         graph_tsne(path, label_to_color, domain_to_marker, dataset_type="train")
#         graph_tsne(path, label_to_color, domain_to_marker, dataset_type="test")

# # ---- graph for looking at different distribution for different data
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
    
#     # change noise to 0
#     if model_hparams['flip_prob'] != None:
#         model_hparams['flip_prob'] = 0
#         model_hparams['study_noise'] = 0
#     print(f"Model hparams after removing noise: {model_hparams}")

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

#     # Choose a target class label to inspect.
#     unique_label = sorted(df["label"].unique())
#     target_label = unique_label[0]

#     # Filter the DataFrame for the target class.
#     sub_df = df[df["label"] == target_label]

#     # Get the unique domains for this class; take up to 4 domains.
#     unique_domains = sorted(sub_df["domain"].unique())[:4]

#     # Create a 2x2 grid of subplots.
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))

#     # Loop through each domain and its corresponding axis.
#     for ax, domain in zip(axes.flatten(), unique_domains):
#         domain_df = sub_df[sub_df["domain"] == domain]
#         sns.scatterplot(
#             data=domain_df,
#             x="x",
#             y="y",
#             ax=ax,
#             s=60,
#             alpha=0.8,
#             edgecolor="k"
#         )
#         ax.set_title(f"Class {target_label} in Domain {domain}")
#         ax.set_xlabel("")
#         ax.set_ylabel("")

#     # If there are fewer than 4 domains, hide the extra subplots.
#     for ax in axes.flatten()[len(unique_domains):]:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(f"results/figures_comparison/domainComp_{args['algorithm']}_{data_name}_tsne_nonoise.png", bbox_inches="tight", dpi=300)


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir_paths = [
#         "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         # "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#         # "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
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
#     """Get unique class labels and domains across all models."""
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

# def graph_tsne(checkpoint_dir_path, dataset_type, ax=None, legend=False):
#     """
#     Generate a t-SNE plot for either train or test dataset of a given model.
#     If `ax` is provided, plot on that subplot; otherwise create a new figure.
#     If `legend=True`, we tell seaborn to produce a full legend so we can capture handles/labels.
#     """
#     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     folder_name = os.path.basename(checkpoint_dir_path)

#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]

#     # Disable noise by setting flip_prob and study_noise to 0
#     if model_hparams.get('flip_prob') is not None:
#         model_hparams['flip_prob'] = 0
#         model_hparams['study_noise'] = 0
#     print(f"Model hparams after removing noise: {model_hparams}")

#     # Load model
#     state_dict = {
#         k: v for k, v in checkpoint["model_dict"].items()
#         if "maplayers" not in k
#     }
#     model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     featurizer = model.featurizer.to(device)

#     # Load dataset
#     data_name = args["dataset"].split("&")[0]
#     dataset_class = vars(datasets)[data_name]
#     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)

#     if torch.cuda.is_available():
#         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#         if free_mem_gb > 20:
#             print("Detected large GPU memory. Setting dataset_obj.N_WORKERS = 1")
#             dataset_obj.N_WORKERS = 0

#     # Split train/test
#     in_splits, out_splits = [], []
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
#         in_splits.append((in_, None))
#         if len(out) > 0:
#             out_splits.append((out, None))

#     if dataset_type == "train":
#         selected_splits = in_splits  # Train dataset
#         eval_loader_names = [f"env{i}_train" for i in range(len(in_splits))]
#     else:
#         selected_splits = out_splits  # Test dataset
#         eval_loader_names = [f"env{i}_test" for i in range(len(out_splits))]

#     eval_loaders = [
#         FastDataLoader(
#             dataset=env,
#             batch_size=model_hparams["test_batch_size"],
#             num_workers=dataset_obj.N_WORKERS
#         )
#         for env, _ in selected_splits
#     ]

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

#     if not all_features:
#         print(f"Skipping {dataset_type} set for {folder_name} (no data available)")
#         return None, None

#     # TSNE
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

#     # Sort your labels/domains so the legend is in ascending order
#     sorted_labels = sorted(df["label"].unique(), key=lambda x: int(x))
#     sorted_domains = sorted(df["domain"].unique(), key=lambda x: int(x))

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))

#     scatter_obj = sns.scatterplot(
#         data=df,
#         x="x",
#         y="y",
#         hue="label",
#         style="domain",
#         hue_order=sorted_labels,     # ensures label legend is in ascending numeric order
#         style_order=sorted_domains,  # ensures domain legend is in ascending numeric order
#         alpha=0.7,
#         legend='full' if legend else False,
#         ax=ax
#     )
#     ax.set_title(f"{args['algorithm']} ({dataset_type} set)")

#     # Make each subplot a square
#     ax.set_aspect("equal", adjustable="box")

#     if legend:
#         handles, labels = ax.get_legend_handles_labels()
#         return handles, labels
#     return None, None


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     checkpoint_dir_paths = [
#         "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
#         "./results/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
#         "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
#     ]

#     num_models = len(checkpoint_dir_paths)
#     # 2 rows × 3 columns => each subplot is about 6×6 => total 18×12
#     fig, axes = plt.subplots(nrows=2, ncols=num_models, figsize=(26, 16))

#     final_handles, final_labels = None, None

#     for col_idx, path in enumerate(checkpoint_dir_paths):
#         # Top row: train set (no legend)
#         graph_tsne(path, dataset_type="train", ax=axes[0, col_idx], legend=False)

#         # Bottom row: test set (legend only on the last subplot)
#         is_last_subplot = (col_idx == num_models - 1)
#         handles, labels = graph_tsne(path, dataset_type="test", ax=axes[1, col_idx], legend=is_last_subplot)

#         if is_last_subplot and handles and labels:
#             final_handles, final_labels = handles, labels

#     # Create a single legend on the right side
#     if final_handles and final_labels:
#         # place it to the right
#         fig.legend(
#             final_handles,
#             final_labels,
#             loc="center left",
#             bbox_to_anchor=(1.02, 0.5),
#             ncol=1  # one column in the legend
#         )

#     fig.suptitle("t-SNE: Train & Test Sets (Each Subplot is Square)", fontsize=16)
#     # Leave room on the right for the legend
#     plt.tight_layout(rect=[0, 0, 0.9, 1])

#     # Save & show
#     plt.savefig("results/figures_comparison/tsne_train_test_six_subplots_single_legend_square.png", dpi=300)
#     # plt.show()


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_unique_labels_domains(checkpoint_dir_paths):
    """Get unique class labels and domains across all models."""
    all_labels_set = set()
    all_domains_set = set()

    for checkpoint_dir_path in checkpoint_dir_paths:
        checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Collect labels
        labels = checkpoint["model_dict"].get("labels", None)
        if labels is not None:
            all_labels_set.update(labels.numpy().tolist())

        # Collect domains
        args = checkpoint["args"]
        data_name = args["dataset"].split("&")[0]
        dataset_class = vars(datasets)[data_name]
        dataset_obj = dataset_class(args["data_dir"], args["test_envs"], checkpoint["model_hparams"])

        for env_i, _ in enumerate(dataset_obj):
            all_domains_set.add(env_i)

    return sorted(all_labels_set), sorted(all_domains_set)

def graph_tsne(checkpoint_dir_path, ax=None, legend=False):
    """
    Generate a t-SNE plot for a given checkpoint using Seaborn.
    - ax: Matplotlib Axes to plot on (no new figure is created if provided).
    - legend: If True, returns legend handles/labels for a single global legend.
    """
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    folder_name = os.path.basename(checkpoint_dir_path)

    input_shape   = checkpoint["model_input_shape"]
    num_classes   = checkpoint["model_num_classes"]
    num_domains   = checkpoint["model_num_domains"]
    model_hparams = checkpoint["model_hparams"]
    args          = checkpoint["args"]

    # Disable noise
    if model_hparams.get('flip_prob') is not None:
        model_hparams['flip_prob'] = 0
        model_hparams['study_noise'] = 0

    # Build model
    state_dict = {k: v for k, v in checkpoint["model_dict"].items() if "maplayers" not in k}
    model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    featurizer = model.featurizer.to(device)

    # Build dataset
    data_name = args["dataset"].split("&")[0]
    dataset_class = vars(datasets)[data_name]
    dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)

    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
        if free_mem_gb > 20:
            dataset_obj.N_WORKERS = 0

    # Split into train/test
    in_splits, out_splits = [], []
    for env_i, env in enumerate(dataset_obj):
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args['holdout_fraction']),
            misc.seed_hash(args['trial_seed'], env_i)
        )
        if env_i in args['test_envs']:
            uda, in_ = misc.split_dataset(
                in_,
                int(len(in_) * args['uda_holdout_fraction']),
                misc.seed_hash(args['trial_seed'], env_i)
            )
        in_splits.append((in_, None))
        if len(out) > 0:
            out_splits.append((out, None))

    # Here we use the train splits (in_splits).
    # If you want test data, swap in_splits -> out_splits
    selected_splits = in_splits
    eval_loader_names = [f"env{i}_train" for i in range(len(in_splits))]

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
        for env, _ in selected_splits
    ]

    # Extract features
    all_features, all_labels, all_domains = [], [], []
    for name, loader in zip(eval_loader_names, eval_loaders):
        match = re.search(r"env(\d+)_", name)
        domain_i = int(match.group(1)) if match else -1
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feats = featurizer(x).cpu()
                all_features.append(feats)
                all_labels.append(y.cpu())
                all_domains.append(torch.full_like(y, domain_i))

    if not all_features:
        print(f"Skipping {folder_name} due to no data")
        return None, None

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels   = torch.cat(all_labels, dim=0).numpy()
    all_domains  = torch.cat(all_domains, dim=0).numpy()

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(all_features)

    # Build a DataFrame
    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label": all_labels.astype(str),
        "domain": all_domains.astype(str)
    })

    # Dynamically create a palette & marker map
    unique_labels_in_data = sorted(df["label"].unique(), key=lambda x: int(x))
    palette_colors = sns.color_palette("tab10", len(unique_labels_in_data))
    label_to_color_map = {lbl: palette_colors[i] for i, lbl in enumerate(unique_labels_in_data)}

    unique_domains_in_data = sorted(df["domain"].unique(), key=lambda x: int(x))
    fixed_markers = ["o", "s", "X", "D", "^", "P", "*", "v", "h", "p"]
    domain_to_marker_map = {
        str(dom): fixed_markers[i % len(fixed_markers)]
        for i, dom in enumerate(unique_domains_in_data)
    }

    # Create Axes if needed
    if ax is None:
        fig, ax = plt.subplots()

    # Scatter plot with bigger marker size (s=150), no axis labels
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        style="domain",
        palette=label_to_color_map,
        markers=domain_to_marker_map,
        alpha=0.7,
        s=150,  # Increase marker size further
        legend='full' if legend else False,
        ax=ax
    )
    legend = ax.legend(markerscale=2, fontsize='large', labelspacing=1.5)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('')

    if legend:
        return ax.get_legend_handles_labels()
    return None, None

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example checkpoint paths - now with 4 paths
    checkpoint_dir_paths = [
        "./results_a3w/erm_pacs_baseline/5763d25571338be34ee9cad8670c7b89",
        "./results_a3w/irm_vlcs/0eff050fbf9ae2b44ae55a40dccf028c",
        "./results_a3w/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044",
        "./results_a3w/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad",
    ]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    axes = axes.flatten()  # Flatten to make indexing easier

    final_handles, final_labels = None, None
    for i, path in enumerate(checkpoint_dir_paths):
        is_last = (i == len(checkpoint_dir_paths) - 1)
        handles, labels = graph_tsne(path, ax=axes[i], legend=is_last)
        if is_last and handles and labels:
            final_handles, final_labels = handles, labels

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("results_a3w/figures_comparison/four_subplots_tsne_2x2_grid.png", dpi=300, bbox_inches='tight')
    plt.show()

