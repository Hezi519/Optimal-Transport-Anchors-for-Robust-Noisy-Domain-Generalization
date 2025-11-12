# # # import os
# # # import re
# # # import torch
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import pandas as pd
# # # from sklearn.manifold import TSNE

# # # # DomainBed imports
# # # from domainbed import algorithms, datasets
# # # from domainbed.lib.fast_data_loader import FastDataLoader
# # # from domainbed.lib import misc

# # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # # def darken_color(base_color, norm_distance, min_factor=0.3):
# # #     """
# # #     Darkens the given base_color based on norm_distance.
# # #     When norm_distance is 0 (point is closest), the color is dark (factor=min_factor).
# # #     When norm_distance is 1, the color remains unchanged.
# # #     """
# # #     factor = min_factor + (1 - min_factor) * (1 - norm_distance)
# # #     return tuple(c * factor for c in base_color)

# # # def graph_embeddings_with_anchors(checkpoint_dir_path, ax=None):
# # #     """
# # #     Generates a TSNE scatterplot showing:
# # #       1) The mapped data embeddings (featurizer output passed through mapping layers)
# # #          from the training splits.
# # #       2) NLP anchors generated using CLIP (one per class) from the dataset’s class names.
# # #          Data points are colored using a base color for their class but darkened according
# # #          to their TSNE distance to their corresponding NLP anchor.
# # #       3) NLP anchors are overlaid using a distinct marker ("P").
# # #     """
# # #     # --- Load checkpoint and model ---
# # #     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
# # #     checkpoint = torch.load(checkpoint_path, map_location="cpu")
# # #     folder_name = os.path.basename(checkpoint_dir_path)
    
# # #     input_shape   = checkpoint["model_input_shape"]
# # #     num_classes   = checkpoint["model_num_classes"]
# # #     num_domains   = checkpoint["model_num_domains"]
# # #     model_hparams = checkpoint["model_hparams"]
# # #     args          = checkpoint["args"]
    
# # #     # Disable noise for visualization
# # #     if model_hparams.get("flip_prob") is not None:
# # #         model_hparams["flip_prob"] = 0
# # #         model_hparams["study_noise"] = 0

# # #     # Load the full state dict (including mapping layers)
# # #     state_dict = checkpoint["model_dict"]  # do not filter out "maplayers"
# # #     model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
# # #     model.load_state_dict(state_dict, strict=False)
# # #     model.eval()
# # #     featurizer = model.featurizer.to("cpu")
# # #     # mapping layers should now be in model.maplayers

# # #     # --- Build dataset and extract mapped features ---
# # #     data_name = args["dataset"].split("&")[0]
# # #     dataset_class = vars(datasets)[data_name]
# # #     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
# # #     if torch.cuda.is_available():
# # #         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
# # #         if free_mem_gb > 20:
# # #             dataset_obj.N_WORKERS = 0

# # #     all_features_list = []
# # #     all_labels_list = []
# # #     # Loop over all environments (training splits)
# # #     for env_i, env in enumerate(dataset_obj):
# # #         out, in_ = misc.split_dataset(
# # #             env,
# # #             int(len(env) * args["holdout_fraction"]),
# # #             misc.seed_hash(args["trial_seed"], env_i)
# # #         )
# # #         loader = FastDataLoader(dataset=in_, batch_size=model_hparams["test_batch_size"],
# # #                                 num_workers=dataset_obj.N_WORKERS)
# # #         with torch.no_grad():
# # #             for x, y in loader:
# # #                 x = x.to("cpu")
# # #                 raw_feats = featurizer(x)  # shape: (batch_size, feature_dim)
# # #                 # Now pass each sample through its corresponding mapping layer:
# # #                 mapped_feats = []
# # #                 for i in range(raw_feats.size(0)):
# # #                     class_idx = y[i].item()  # get label as int
# # #                     # Apply the mapping layer for that class.
# # #                     mapped_feat = model.maplayers[class_idx](raw_feats[i].unsqueeze(0))
# # #                     mapped_feats.append(mapped_feat)
# # #                 mapped_feats = torch.cat(mapped_feats, dim=0)  # shape: (batch_size, mapped_feature_dim)
# # #                 all_features_list.append(mapped_feats)
# # #                 all_labels_list.append(y)
# # #     if not all_features_list:
# # #         print("No data available!")
# # #         return
# # #     all_features = torch.cat(all_features_list, dim=0).numpy()
# # #     all_labels = torch.cat(all_labels_list, dim=0).numpy()  # assume labels are integers

# # #     # --- Generate NLP anchors using CLIP ---
# # #     # Use CLIP to compute text features for each class.
# # #     import clip
# # #     device_clip = "cpu"  # Change to "cuda" if available and desired.
# # #     clip_model, clip_preprocess = clip.load("RN50", device_clip)
# # #     try:
# # #         # Try to get class names from the dataset (assuming the first sub-dataset has attribute 'classes')
# # #         classes_names = dataset_obj.datasets[0].classes
# # #     except AttributeError:
# # #         print("Could not retrieve class names from dataset; using placeholders.")
# # #         classes_names = [str(i) for i in range(num_classes)]
    
# # #     # Generate one prompt per class.
# # #     text_prompts = [[f"a photo of a {item}"] for item in classes_names]
# # #     all_text_features = []
# # #     for prompts in text_prompts:
# # #         text_tokens = clip.tokenize(prompts).to(device_clip)
# # #         with torch.no_grad():
# # #             text_features = clip_model.encode_text(text_tokens)
# # #             text_features_avg = text_features.mean(dim=0)
# # #         all_text_features.append(text_features_avg)
# # #     anchors = torch.stack(all_text_features)  # shape: (num_classes, feature_dim)
# # #     anchors_np = anchors.detach().cpu().numpy()
    
# # #     # --- TSNE: Run TSNE on both data points and anchors ---
# # #     combined = np.concatenate([all_features, anchors_np], axis=0)
# # #     tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
# # #     N = all_features.shape[0]
# # #     data_tsne = tsne_all[:N]
# # #     anchors_tsne = tsne_all[N:]
    
# # #     # --- Build DataFrame for data points ---
# # #     df = pd.DataFrame({
# # #         "x": data_tsne[:, 0],
# # #         "y": data_tsne[:, 1],
# # #         "label": all_labels.astype(str)
# # #     })
    
# # #     # Set base colors for each class.
# # #     unique_labels = sorted(list(set(df["label"])))
# # #     palette = sns.color_palette("tab10", len(unique_labels))
# # #     base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    
# # #     # Compute TSNE distances from data points to their class’s anchor.
# # #     distances = []
# # #     for idx, row in df.iterrows():
# # #         lbl = row["label"]
# # #         class_idx = int(lbl)  # assume labels are "0", "1", ...
# # #         anchor_coord = anchors_tsne[class_idx]
# # #         point_coord = np.array([row["x"], row["y"]])
# # #         dist = np.linalg.norm(point_coord - anchor_coord)
# # #         distances.append(dist)
# # #     df["dist"] = distances
    
# # #     # Normalize distances per class and compute a final color for each data point.
# # #     final_colors = []
# # #     for lbl in unique_labels:
# # #         mask = df["label"] == lbl
# # #         d_vals = df.loc[mask, "dist"].values
# # #         if len(d_vals) > 1:
# # #             d_min, d_max = d_vals.min(), d_vals.max()
# # #             norm = (d_vals - d_min) / (d_max - d_min)
# # #         else:
# # #             norm = np.zeros_like(d_vals)
# # #         base_color = base_color_map[lbl]
# # #         colors_for_class = [darken_color(base_color, nd, min_factor=0.3) for nd in norm]
# # #         final_colors.extend(colors_for_class)
# # #     df["final_color"] = final_colors
    
# # #     # --- Plotting ---
# # #     if ax is None:
# # #         fig, ax = plt.subplots(figsize=(10,8))
    
# # #     # Plot data points for each class.
# # #     for lbl in unique_labels:
# # #         sub_df = df[df["label"] == lbl]
# # #         ax.scatter(sub_df["x"], sub_df["y"], c=list(sub_df["final_color"]),
# # #                    label=f"Class {lbl}", s=50, alpha=0.8)
    
# # #     # Overlay NLP anchors with a distinct marker ("P").
# # #     for lbl in unique_labels:
# # #         anchor_coord = anchors_tsne[int(lbl)]
# # #         ax.scatter(anchor_coord[0], anchor_coord[1], marker="P", s=300,
# # #                    color=base_color_map[lbl], edgecolor="black", linewidth=2,
# # #                    label=f"Anchor {lbl}")
    
# # #     ax.set_xlabel("")
# # #     ax.set_ylabel("")
# # #     ax.set_xticks([])
# # #     ax.set_yticks([])
# # #     ax.set_title("Embeddings with NLP Anchors")
    
# # #     ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# # #     plt.tight_layout(rect=[0,0,0.85,1])
# # #     plt.savefig("./results/anchor_plot.png")

# # # if __name__ == "__main__":
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     # Set the checkpoint directory (must include "model_best.pkl" with mapping layers)
# # #     checkpoint_dir = "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad"
# # #     graph_embeddings_with_anchors(checkpoint_dir)


# # import os
# # import re
# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import pandas as pd
# # from sklearn.manifold import TSNE
# # import torch.nn as nn

# # # DomainBed imports
# # from domainbed import algorithms, datasets
# # from domainbed.lib.fast_data_loader import FastDataLoader
# # from domainbed.lib import misc

# # os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# # def darken_color(base_color, norm_distance, min_factor=0.3):
# #     """
# #     Darkens the given base_color based on norm_distance.
# #     When norm_distance is 0 (point is closest), factor = min_factor (darker);
# #     when norm_distance is 1, factor = 1 (original brightness).
# #     """
# #     factor = min_factor + (1 - min_factor) * (1 - norm_distance)
# #     return tuple(c * factor for c in base_color)

# # def graph_embeddings_with_anchors(checkpoint_dir_path, ax=None):
# #     """
# #     Generates a TSNE scatterplot showing:
# #       1) Mapped data embeddings: The output of the featurizer is passed through
# #          the mapping layers stored in the checkpoint.
# #       2) NLP anchors generated using CLIP (one per class) based on text prompts.
# #          Data points are colored by class with darker colors if they are closer
# #          (in TSNE space) to their corresponding NLP anchor.
# #       3) NLP anchors are overlaid with a distinct marker ("P").
# #     """
# #     # --- Load checkpoint and model settings ---
# #     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
# #     checkpoint = torch.load(checkpoint_path, map_location="cpu")
# #     folder_name = os.path.basename(checkpoint_dir_path)
    
# #     input_shape   = checkpoint["model_input_shape"]
# #     num_classes   = checkpoint["model_num_classes"]
# #     num_domains   = checkpoint["model_num_domains"]
# #     model_hparams = checkpoint["model_hparams"]
# #     args          = checkpoint["args"]
    
# #     # Disable noise for visualization
# #     if model_hparams.get("flip_prob") is not None:
# #         model_hparams["flip_prob"] = 0
# #         model_hparams["study_noise"] = 0

# #     # --- Build model (mapping layers are loaded from the checkpoint) ---
# #     state_dict = checkpoint["model_dict"]  # Contains keys like "maplayers.0.weight", etc.
# #     model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
# #     model.load_state_dict(state_dict, strict=False)
# #     model.eval()
# #     featurizer = model.featurizer.to("cpu")
# #     # We assume that model.maplayers is stored in the checkpoint and loaded by the model.
    
# #     # --- Build dataset and extract mapped features ---
# #     data_name = args["dataset"].split("&")[0]
# #     dataset_class = vars(datasets)[data_name]
# #     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
# #     if torch.cuda.is_available():
# #         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
# #         if free_mem_gb > 20:
# #             dataset_obj.N_WORKERS = 0

# #     all_features_list = []
# #     all_labels_list = []
    
# #     model.set_nlp_anchor(dataset_obj)
# #     nlpanchors = model.nlpanchor
# #     print(nlpanchors.shape) #torch.Size([7, 1024])
# #     anchors_np = nlpanchors.detach().cpu().numpy()
    
# #     for env_i, env in enumerate(dataset_obj):
# #         out, in_ = misc.split_dataset(
# #             env,
# #             int(len(env) * args["holdout_fraction"]),
# #             misc.seed_hash(args["trial_seed"], env_i)
# #         )
# #         loader = FastDataLoader(dataset=in_, batch_size=model_hparams["test_batch_size"],
# #                                 num_workers=dataset_obj.N_WORKERS)
# #         with torch.no_grad():
# #             for x, y in loader:
# #                 x = x.to("cpu")
# #                 raw_feats = featurizer(x)  # shape: (batch, feat_dim)
# #                 mapped_feats = []
# #                 # Pass each feature through the corresponding mapping layer.
# #                 for i in range(raw_feats.size(0)):
# #                     class_idx = y[i].item()
# #                     # Use the stored mapping layer for this class.
# #                     mapped_feat = model.maplayers[class_idx](raw_feats[i].unsqueeze(0))
# #                     mapped_feats.append(mapped_feat)
# #                 mapped_feats = torch.cat(mapped_feats, dim=0)
# #                 all_features_list.append(mapped_feats)
# #                 all_labels_list.append(y)
# #     if not all_features_list:
# #         print("No data available!")
# #         return
# #     all_features = torch.cat(all_features_list, dim=0).numpy()  # shape: (N, mapped_dim)
# #     all_labels = torch.cat(all_labels_list, dim=0).numpy()        # shape: (N,)
    
# #     np.save("graph/all_features.npy", all_features)
# #     np.save("graph/all_labels.npy", all_labels)
# #     np.save("graph/anchors.npy", anchors_np)

    
# #     # --- TSNE: Combine mapped features and anchors ---
# #     combined = np.concatenate([all_features, anchors_np], axis=0)
# #     tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
# #     N = all_features.shape[0]
# #     data_tsne = tsne_all[:N]
# #     anchors_tsne = tsne_all[N:]
    
# #     # --- Build a DataFrame for data points ---
# #     df = pd.DataFrame({
# #         "x": data_tsne[:, 0],
# #         "y": data_tsne[:, 1],
# #         "label": all_labels.astype(str)
# #     })
    
# #     # --- Set base colors for each class ---
# #     unique_labels = sorted(list(set(df["label"])))
# #     palette = sns.color_palette("tab10", len(unique_labels))
# #     base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    
# #     # --- Compute TSNE distances from data points to their corresponding class anchor ---
# #     distances = []
# #     for idx, row in df.iterrows():
# #         lbl = row["label"]
# #         class_idx = int(lbl)  # assumes labels are "0", "1", etc.
# #         anchor_coord = anchors_tsne[class_idx]
# #         point_coord = np.array([row["x"], row["y"]])
# #         dist = np.linalg.norm(point_coord - anchor_coord)
# #         distances.append(dist)
# #     df["dist"] = distances
    
# #     # --- Normalize distances per class and compute final color (darker if closer to anchor) ---
# #     final_colors = []
# #     for lbl in unique_labels:
# #         mask = df["label"] == lbl
# #         d_vals = df.loc[mask, "dist"].values
# #         if len(d_vals) > 1:
# #             d_min, d_max = d_vals.min(), d_vals.max()
# #             norm = (d_vals - d_min) / (d_max - d_min)
# #         else:
# #             norm = np.zeros_like(d_vals)
# #         base_color = base_color_map[lbl]
# #         colors_for_class = [darken_color(base_color, nd, min_factor=0.3) for nd in norm]
# #         final_colors.extend(colors_for_class)
# #     df["final_color"] = final_colors
    
# #     # --- Plotting ---
# #     if ax is None:
# #         fig, ax = plt.subplots(figsize=(10, 8))
    
# #     # Plot data points for each class
# #     for lbl in unique_labels:
# #         sub_df = df[df["label"] == lbl]
# #         ax.scatter(sub_df["x"], sub_df["y"], c=list(sub_df["final_color"]),
# #                    label=f"Class {lbl}", s=50, alpha=0.8)
    
# #     # # Overlay NLP anchors with marker "P" and large size.
# #     # for lbl in unique_labels:
# #     #     anchor_coord = anchors_tsne[int(lbl)]
# #     #     ax.scatter(anchor_coord[0], anchor_coord[1], marker="P", s=300,
# #     #                color=base_color_map[lbl], edgecolor="black", linewidth=2,
# #     #                label=f"Anchor {lbl}")
    
# #     ax.set_xlabel("")
# #     ax.set_ylabel("")
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# #     ax.set_title("Embeddings with NLP Anchors")
    
# #     ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# #     plt.tight_layout(rect=[0, 0, 0.85, 1])
# #     plt.savefig("./results/figures_comparison/anchor_plot_v2.png")

# # if __name__ == "__main__":
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     checkpoint_dir = "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad"
# #     graph_embeddings_with_anchors(checkpoint_dir)

# import os
# import re
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.manifold import TSNE

# # DomainBed imports
# from domainbed import algorithms, datasets
# from domainbed.lib.fast_data_loader import FastDataLoader
# from domainbed.lib import misc

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def darken_color(base_color, norm_distance, min_factor=0.3):
#     """
#     Darkens the given base_color based on norm_distance.
#     When norm_distance is 0 (point is closest), factor = min_factor (darker);
#     when norm_distance is 1, factor = 1 (original brightness).
#     """
#     factor = min_factor + (1 - min_factor) * (1 - norm_distance)
#     return tuple(c * factor for c in base_color)

# def cosine_similarity(u, v):
#     """Compute the cosine similarity between two vectors u and v."""
#     return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# def graph_embeddings_with_anchors(checkpoint_dir_path, ax=None):
#     """
#     Generates a TSNE scatterplot showing:
#       1) Mapped data embeddings (featurizer output passed through mapping layers)
#          from the training splits.
#       2) NLP anchors generated using CLIP (one per class) based on text prompts.
#          Data points are colored by class with darker colors if they are closer
#          (in TSNE space) to their corresponding NLP anchor.
#       3) NLP anchors are overlaid with a distinct marker ("P").
#     Additionally, we compute cosine similarity (instead of Euclidean distance)
#     between each data point and its class anchor, so that we can later create a bar plot.
#     """
#     --- Load checkpoint and model settings ---
#     checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     folder_name = os.path.basename(checkpoint_dir_path)
    
#     input_shape   = checkpoint["model_input_shape"]
#     num_classes   = checkpoint["model_num_classes"]
#     num_domains   = checkpoint["model_num_domains"]
#     model_hparams = checkpoint["model_hparams"]
#     args          = checkpoint["args"]
    
#     # Disable noise for visualization.
#     if model_hparams.get("flip_prob") is not None:
#         model_hparams["flip_prob"] = 0
#         model_hparams["study_noise"] = 0

#     # --- Build model (mapping layers are stored in the checkpoint) ---
#     state_dict = checkpoint["model_dict"]  # Contains keys like "maplayers.0.weight", etc.
#     model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     featurizer = model.featurizer.to("cpu")
    
#     # --- Build dataset and extract mapped features ---
#     data_name = args["dataset"].split("&")[0]
#     dataset_class = vars(datasets)[data_name]
#     dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
#     if torch.cuda.is_available():
#         free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#         if free_mem_gb > 20:
#             dataset_obj.N_WORKERS = 0

#     all_features_list = []
#     all_labels_list = []
    
#     # Generate NLP anchors and mapping layers by calling set_nlp_anchor.
#     model.set_nlp_anchor(dataset_obj)
#     nlpanchors = model.nlpanchor  # shape: (num_classes, anchor_dim)
#     anchors_np = nlpanchors.detach().cpu().numpy()
#     print("NLP anchors shape:", anchors_np.shape)
    
#     for env_i, env in enumerate(dataset_obj):
#         out, in_ = misc.split_dataset(
#             env,
#             int(len(env) * args["holdout_fraction"]),
#             misc.seed_hash(args["trial_seed"], env_i)
#         )
#         loader = FastDataLoader(dataset=in_, batch_size=model_hparams["test_batch_size"],
#                                 num_workers=dataset_obj.N_WORKERS)
#         with torch.no_grad():
#             for x, y in loader:
#                 x = x.to("cpu")
#                 raw_feats = featurizer(x)  # shape: (batch, feat_dim)
#                 mapped_feats = []
#                 # Pass each feature through its corresponding mapping layer.
#                 for i in range(raw_feats.size(0)):
#                     class_idx = y[i].item()
#                     mapped_feat = model.maplayers[class_idx](raw_feats[i].unsqueeze(0))
#                     mapped_feats.append(mapped_feat)
#                 mapped_feats = torch.cat(mapped_feats, dim=0)
#                 all_features_list.append(mapped_feats)
#                 all_labels_list.append(y)
#     if not all_features_list:
#         print("No data available!")
#         return None, None
#     all_features = torch.cat(all_features_list, dim=0).numpy()  # shape: (N, mapped_dim)
#     all_labels = torch.cat(all_labels_list, dim=0).numpy()        # shape: (N,)
    
#     # Optionally, save features, labels, and anchors.
#     np.save("graph/all_features.npy", all_features)
#     np.save("graph/all_labels.npy", all_labels)
#     np.save("graph/anchors.npy", anchors_np)
    
#     # --- TSNE: Combine mapped features and anchors ---
#     combined = np.concatenate([all_features, anchors_np], axis=0)
#     tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
#     N = all_features.shape[0]
#     data_tsne = tsne_all[:N]
#     anchors_tsne = tsne_all[N:]
    
#     # --- Build DataFrame for data points ---
#     df = pd.DataFrame({
#         "x": data_tsne[:, 0],
#         "y": data_tsne[:, 1],
#         "label": all_labels.astype(str)
#     })
    
#     # --- Set base colors for each class ---
#     unique_labels = sorted(list(set(df["label"])))
#     palette = sns.color_palette("tab10", len(unique_labels))
#     base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    
#     # --- Compute cosine similarity from each data point to its corresponding class anchor ---
#     cos_sims = []
#     for idx, row in df.iterrows():
#         lbl = row["label"]
#         class_idx = int(lbl)  # assumes labels are "0", "1", ...
#         anchor_coord = anchors_tsne[class_idx]
#         point_coord = np.array([row["x"], row["y"]])
#         cos_sim = cosine_similarity(point_coord, anchor_coord)
#         cos_sims.append(cos_sim)
#     df["cos_sim"] = cos_sims
    
#     # --- For visualization: also compute final colors (as before) if desired ---
#     distances = []
#     for idx, row in df.iterrows():
#         lbl = row["label"]
#         class_idx = int(lbl)
#         anchor_coord = anchors_tsne[class_idx]
#         point_coord = np.array([row["x"], row["y"]])
#         dist = np.linalg.norm(point_coord - anchor_coord)
#         distances.append(dist)
#     df["dist"] = distances
#     final_colors = []
#     for lbl in unique_labels:
#         mask = df["label"] == lbl
#         d_vals = df.loc[mask, "dist"].values
#         if len(d_vals) > 1:
#             d_min, d_max = d_vals.min(), d_vals.max()
#             norm = (d_vals - d_min) / (d_max - d_min)
#         else:
#             norm = np.zeros_like(d_vals)
#         base_color = base_color_map[lbl]
#         colors_for_class = [darken_color(base_color, nd, min_factor=0.3) for nd in norm]
#         final_colors.extend(colors_for_class)
#     df["final_color"] = final_colors
    
#     # --- Plotting (for TSNE visualization) ---
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 8))
    
#     for lbl in unique_labels:
#         sub_df = df[df["label"] == lbl]
#         ax.scatter(sub_df["x"], sub_df["y"], c=list(sub_df["final_color"]),
#                    label=f"Class {lbl}", s=50, alpha=0.8)
    
#     for lbl in unique_labels:
#         anchor_coord = anchors_tsne[int(lbl)]
#         ax.scatter(anchor_coord[0], anchor_coord[1], marker="P", s=300,
#                    color=base_color_map[lbl], edgecolor="black", linewidth=2,
#                    label=f"Anchor {lbl}")
    
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title("Embeddings with NLP Anchors")
    
#     ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     plt.savefig("./results/figures_comparison/anchor_plot.png", dpi=300)
#     plt.show()
    
#     # --- Now, produce a bar plot of the average cosine similarity per class ---
#     avg_cos_sim = df.groupby("label")["cos_sim"].mean().reset_index()
#     avg_cos_sim["label"] = avg_cos_sim["label"].astype(int)
#     avg_cos_sim = avg_cos_sim.sort_values("label")
    
#     plt.figure(figsize=(8,6))
#     sns.barplot(data=avg_cos_sim, x="label", y="cos_sim", palette=palette)
#     plt.xlabel("Class")
#     plt.ylabel("Average Cosine Similarity")
#     plt.title("Average Cosine Similarity to NLP Anchor per Class")
#     plt.tight_layout()
#     plt.savefig("./results/figures_comparison/anchor_bar_plot_cosine_similarity.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     checkpoint_dir = "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad"
#     graph_embeddings_with_anchors(checkpoint_dir)

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE

# def cosine_similarity(u, v):
#     """Compute cosine similarity between two vectors u and v."""
#     return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# # --- Load saved features, labels, and anchors ---
# # Adjust these paths as needed.
# features_path = "graph/all_features.npy"
# labels_path   = "graph/all_labels.npy"
# anchors_path  = "graph/anchors.npy"

# all_features = np.load(features_path)  # shape: (N, mapped_dim)
# all_labels   = np.load(labels_path)    # shape: (N,)
# anchors_np   = np.load(anchors_path)    # shape: (num_classes, anchor_dim)

# print("Features shape:", all_features.shape)
# print("Labels shape:", all_labels.shape)
# print("Anchors shape:", anchors_np.shape)

# # --- Run TSNE on the combined data (features and anchors) ---
# combined = np.concatenate([all_features, anchors_np], axis=0)
# tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
# N = all_features.shape[0]
# data_tsne = tsne_all[:N]
# anchors_tsne = tsne_all[N:]

# # --- Build DataFrame for data points ---
# df = pd.DataFrame({
#     "x": data_tsne[:, 0],
#     "y": data_tsne[:, 1],
#     "label": all_labels.astype(str)
# })

# # --- Set base colors (optional, for visualization) ---
# unique_labels = sorted(list(set(df["label"])))
# palette = sns.color_palette("tab10", len(unique_labels))
# base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

# # --- Compute cosine distance (1 - cosine similarity) in TSNE space ---
# cos_sims = []
# for idx, row in df.iterrows():
#     lbl = row["label"]
#     class_idx = int(lbl)  # assumes labels are "0", "1", etc.
#     anchor_coord = anchors_tsne[class_idx]
#     point_coord = np.array([row["x"], row["y"]])
#     sim = cosine_similarity(point_coord, anchor_coord)
#     cos_dist = 1 - sim
#     cos_sims.append(cos_dist)
# df["cos_dist"] = cos_sims

# # --- Aggregate per class: compute average intra-class and inter-class cosine distances ---
# records = []
# for lbl in unique_labels:
#     class_idx = int(lbl)
#     anchor = anchors_tsne[class_idx]
#     # Intra-class: select data points with label == lbl
#     pts_intra = df[df["label"] == lbl][["x", "y"]].values
#     if len(pts_intra) > 0:
#         intra_dists = [1 - cosine_similarity(pt, anchor) for pt in pts_intra]
#         avg_intra = np.mean(intra_dists)
#     else:
#         avg_intra = np.nan
#     # Inter-class: select data points with label != lbl
#     pts_inter = df[df["label"] != lbl][["x", "y"]].values
#     if len(pts_inter) > 0:
#         inter_dists = [1 - cosine_similarity(pt, anchor) for pt in pts_inter]
#         avg_inter = np.mean(inter_dists)
#     else:
#         avg_inter = np.nan
#     records.append({"Class": lbl, "Type": "Intra", "Cosine Distance": avg_intra})
#     records.append({"Class": lbl, "Type": "Inter", "Cosine Distance": avg_inter})

# bar_df = pd.DataFrame(records)
# bar_df["Class"] = bar_df["Class"].astype(int)
# bar_df = bar_df.sort_values("Class")

# # --- Plot a grouped bar plot ---
# plt.figure(figsize=(10, 6))
# sns.barplot(data=bar_df, x="Class", y="Cosine Distance", hue="Type", palette="viridis")
# plt.xlabel("Class")
# plt.ylabel("Average Cosine Distance")
# plt.title("Average Cosine Distance to NLP Anchor per Class\n(Intra vs. Inter)")
# plt.tight_layout()
# plt.savefig("./results/figures/bar_plot_cosine_distance_from_saved.png", dpi=300)
# plt.show()

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE

# def cosine_similarity(u, v):
#     """Compute cosine similarity between two vectors."""
#     return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# # --- 1. Load saved features, labels, and anchors ---
# features_path = "graph/all_features.npy"  # shape: (N, mapped_dim)
# labels_path   = "graph/all_labels.npy"    # shape: (N,)
# anchors_path  = "graph/anchors.npy"        # shape: (num_classes, anchor_dim)

# all_features = np.load(features_path)
# all_labels   = np.load(labels_path)
# anchors_np   = np.load(anchors_path)

# print("Features shape:", all_features.shape)
# print("Labels shape:", all_labels.shape)
# print("Anchors shape:", anchors_np.shape)

# # --- 2. Run TSNE on the combined features and anchors ---
# combined = np.concatenate([all_features, anchors_np], axis=0)
# tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
# N = all_features.shape[0]
# data_tsne = tsne_all[:N]
# anchors_tsne = tsne_all[N:]

# # --- 3. Build a DataFrame for data points ---
# df = pd.DataFrame({
#     "x": data_tsne[:, 0],
#     "y": data_tsne[:, 1],
#     "label": all_labels.astype(str)
# })

# # --- 4. Compute distances from each data point to its corresponding class anchor ---
# cos_dists = []
# eucl_dists = []
# for idx, row in df.iterrows():
#     lbl = row["label"]
#     class_idx = int(lbl)  # assumes labels are "0", "1", etc.
#     anchor_coord = anchors_tsne[class_idx]
#     point_coord = np.array([row["x"], row["y"]])
#     # Cosine similarity -> cosine distance:
#     sim = cosine_similarity(point_coord, anchor_coord)
#     cos_dists.append(1 - sim)
#     # Euclidean distance:
#     eucl_dists.append(np.linalg.norm(point_coord - anchor_coord))
# df["cos_dist"] = cos_dists
# df["eucl_dist"] = eucl_dists

# # --- 5. Aggregate metrics per class ---
# records = []
# unique_labels = sorted(list(set(df["label"])), key=lambda x: int(x))
# for lbl in unique_labels:
#     class_idx = int(lbl)
#     # Get the anchor for this class (TSNE coordinate)
#     anchor = anchors_tsne[class_idx]
    
#     # Intra-class: data points with label == lbl.
#     pts_intra = df[df["label"] == lbl][["x", "y"]].values
#     if len(pts_intra) > 0:
#         avg_cos_intra = np.mean([1 - cosine_similarity(pt, anchor) for pt in pts_intra])
#         avg_eucl_intra = np.mean([np.linalg.norm(pt - anchor) for pt in pts_intra])
#     else:
#         avg_cos_intra = np.nan
#         avg_eucl_intra = np.nan

#     # Inter-class: data points with label != lbl.
#     pts_inter = df[df["label"] != lbl][["x", "y"]].values
#     if len(pts_inter) > 0:
#         avg_cos_inter = np.mean([1 - cosine_similarity(pt, anchor) for pt in pts_inter])
#         avg_eucl_inter = np.mean([np.linalg.norm(pt - anchor) for pt in pts_inter])
#     else:
#         avg_cos_inter = np.nan
#         avg_eucl_inter = np.nan

#     records.append({"Class": lbl, "Metric": "Intra Cosine", "Distance": avg_cos_intra})
#     records.append({"Class": lbl, "Metric": "Inter Cosine", "Distance": avg_cos_inter})
#     records.append({"Class": lbl, "Metric": "Intra Euclidean", "Distance": avg_eucl_intra})
#     records.append({"Class": lbl, "Metric": "Inter Euclidean", "Distance": avg_eucl_inter})

# bar_df = pd.DataFrame(records)
# bar_df["Class"] = bar_df["Class"].astype(int)
# bar_df = bar_df.sort_values("Class")

# # --- 6. Plot grouped bar plot ---
# plt.figure(figsize=(10, 6))
# sns.barplot(data=bar_df, x="Class", y="Distance", hue="Metric", palette="viridis")
# plt.xlabel("Class")
# plt.ylabel("Average Distance")
# plt.title("Average Distance from Data Points to NLP Anchor per Class\n(Intra vs. Inter, Cosine & Euclidean)")
# plt.tight_layout()
# plt.savefig("./results/figures/bar_plot_combined_distances.png", dpi=300)
# plt.show()


import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import torch.nn as nn

# DomainBed imports
from domainbed import algorithms, datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

def darken_color(base_color, norm_distance, min_factor=0.3):
    """
    Darkens the given base_color based on norm_distance.
    When norm_distance is 0 (point is closest), factor = min_factor (darker);
    when norm_distance is 1, factor = 1 (original brightness).
    """
    factor = min_factor + (1 - min_factor) * (1 - norm_distance)
    return tuple(c * factor for c in base_color)

def cosine_similarity(u, v):
    """Compute cosine similarity between two vectors u and v."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def graph_tsne(checkpoint_dir_path, ax=None, legend=False):
    """
    Generates a TSNE scatterplot showing:
      1) Mapped data embeddings (featurizer output passed through mapping layers)
         from the training splits.
      2) NLP anchors generated using CLIP (one per class) based on text prompts.
         Data points are colored by class with darker colors if they are closer
         (in TSNE space) to their corresponding NLP anchor.
      3) NLP anchors are overlaid with a distinct marker ("P").
    Domain information is removed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load checkpoint and model settings ---
    checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    folder_name = os.path.basename(checkpoint_dir_path)
    
    input_shape   = checkpoint["model_input_shape"]
    num_classes   = checkpoint["model_num_classes"]
    num_domains   = checkpoint["model_num_domains"]  # not used now
    model_hparams = checkpoint["model_hparams"]
    args          = checkpoint["args"]
    
    # Disable noise for visualization.
    if model_hparams.get('flip_prob') is not None:
        model_hparams['flip_prob'] = 0
        model_hparams['study_noise'] = 0

    # --- Build model (mapping layers are stored in the checkpoint) ---
    state_dict = checkpoint["model_dict"]  # contains keys like "maplayers.0.weight", etc.
    model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    featurizer = model.featurizer.to(device)
    
    # --- Build dataset and extract mapped features ---
    data_name = args["dataset"].split("&")[0]
    dataset_class = vars(datasets)[data_name]
    dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
        if free_mem_gb > 20:
            dataset_obj.N_WORKERS = 0

    all_features_list = []
    all_labels_list = []
    
    # Generate NLP anchors and mapping layers by calling set_nlp_anchor.
    model.set_nlp_anchor(dataset_obj)
    nlpanchors = model.nlpanchor  # shape: (num_classes, anchor_dim)
    anchors_np = nlpanchors.detach().cpu().numpy()
    print("NLP anchors shape:", anchors_np.shape)
    
    # (Optionally load mapping layers from checkpoint state dict)
    map_state_dict = {k: v for k, v in checkpoint["model_dict"].items() if "maplayers" in k}
    maplayers = nn.ModuleList([nn.Linear(featurizer.n_outputs, len(anchors_np[0])).to(device) 
                               for _ in range(num_classes)])
    maplayers.load_state_dict(map_state_dict, strict=False)
    maplayers.eval()
    # We'll use model.maplayers if available, otherwise use the loaded maplayers.
    if hasattr(model, "maplayers"):
        mapping_layers = model.maplayers
    else:
        mapping_layers = maplayers

    eval_loader_names = [f"env{i}_train" for i in range(len(dataset_obj))]
    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=model_hparams["test_batch_size"], num_workers=dataset_obj.N_WORKERS)
        for env in [misc.split_dataset(env, int(len(env)*args["holdout_fraction"]), misc.seed_hash(args["trial_seed"], i))[1]
                    for i, env in enumerate(dataset_obj)]
    ]
    
    for name, loader in zip(eval_loader_names, eval_loaders):
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                raw_feats = featurizer(x)  # shape: (batch, feat_dim)
                mapped_feats = []
                for i in range(raw_feats.size(0)):
                    class_idx = y[i].item()
                    mapped_feat = mapping_layers[class_idx](raw_feats[i].unsqueeze(0))
                    mapped_feats.append(mapped_feat)
                mapped_feats = torch.cat(mapped_feats, dim=0)
                all_features_list.append(mapped_feats)
                all_labels_list.append(y)
    if not all_features_list:
        print(f"Skipping {folder_name} due to no data")
        return None
    all_features = torch.cat(all_features_list, dim=0).detach().cpu().numpy()  # shape: (N, mapped_dim)
    all_labels   = torch.cat(all_labels_list, dim=0).detach().cpu().numpy()        # shape: (N,)
    
    # --- Save features (optional) ---
    np.save("graph/all_features.npy", all_features)
    np.save("graph/all_labels.npy", all_labels)
    np.save("graph/anchors.npy", anchors_np)
    
    # --- Run TSNE on combined features and anchors ---
    combined_features = np.concatenate([all_features, anchors_np], axis=0)
    tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined_features)
    N = all_features.shape[0]
    data_2d = tsne_all[:N]
    anchors_2d = tsne_all[N:]
    
    # --- Build DataFrame for data points ---
    df = pd.DataFrame({
        "x": data_2d[:, 0],
        "y": data_2d[:, 1],
        "label": all_labels.astype(str)
    })
    
    # --- Set base colors for each class ---
    unique_labels = sorted(list(set(df["label"])), key=lambda x: int(x))
    palette = sns.color_palette("tab10", len(unique_labels))
    base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    
    # --- Compute cosine similarity (distance) from each data point to its corresponding NLP anchor ---
    cos_dists = []
    for idx, row in df.iterrows():
        lbl = row["label"]
        class_idx = int(lbl)  # assume labels are "0", "1", etc.
        anchor_coord = anchors_2d[class_idx]
        point_coord = np.array([row["x"], row["y"]])
        sim = cosine_similarity(point_coord, anchor_coord)
        cos_dists.append(1 - sim)
    df["cos_dist"] = cos_dists
    
    # --- (Optional) Compute final colors based on Euclidean distance (for TSNE visualization) ---
    distances = []
    for idx, row in df.iterrows():
        lbl = row["label"]
        class_idx = int(lbl)
        anchor_coord = anchors_2d[class_idx]
        point_coord = np.array([row["x"], row["y"]])
        dist = np.linalg.norm(point_coord - anchor_coord)
        distances.append(dist)
    df["eucl_dist"] = distances
    final_colors = []
    for lbl in unique_labels:
        mask = df["label"] == lbl
        d_vals = df.loc[mask, "eucl_dist"].values
        if len(d_vals) > 1:
            d_min, d_max = d_vals.min(), d_vals.max()
            norm = (d_vals - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(d_vals)
        base_color = base_color_map[lbl]
        colors_for_class = [darken_color(base_color, nd, min_factor=0.3) for nd in norm]
        final_colors.extend(colors_for_class)
    df["final_color"] = final_colors
    
    # --- Plot TSNE visualization (if desired) ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=base_color_map,
        s=150,
        legend='full' if legend else False,
        ax=ax
    )
    if legend:
        ax.legend(markerscale=2, fontsize='large', labelspacing=1.5)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE visualization with NLP anchors")
    
    # Overlay NLP anchors with marker "P"
    for i, anchor_coord in enumerate(anchors_2d):
        ax.scatter(anchor_coord[0], anchor_coord[1],
                   marker="P", s=300,
                   color=base_color_map.get(str(i), "black"),
                   edgecolor="k", linewidth=1.5,
                   label=f"Anchor {i}" if legend else None)
    
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("results/figures_comparison/tsne_nlpgerm_checkpoint.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, anchors_2d, unique_labels, base_color_map

def bar_plot_cosine_distances_from_saved():
    """
    Loads saved features, labels, and anchors, runs TSNE,
    computes average cosine distance (1 - cosine similarity) from each class's NLP anchor
    to data points:
      - Intra-class: data points belonging to the class.
      - Inter-class: data points not belonging to the class.
    Plots a grouped bar plot for each class.
    """
    # Load saved features
    all_features = np.load("graph/all_features.npy")
    all_labels   = np.load("graph/all_labels.npy")
    anchors_np   = np.load("graph/anchors.npy")
    
    # Run TSNE on combined features and anchors.
    combined = np.concatenate([all_features, anchors_np], axis=0)
    tsne_all = TSNE(n_components=2, random_state=0).fit_transform(combined)
    N = all_features.shape[0]
    data_2d = tsne_all[:N]
    anchors_2d = tsne_all[N:]
    
    # Build DataFrame for data points.
    df = pd.DataFrame({
        "x": data_2d[:, 0],
        "y": data_2d[:, 1],
        "label": all_labels.astype(str)
    })
    
    # Set base colors.
    unique_labels = sorted(list(set(df["label"])), key=lambda x: int(x))
    palette = sns.color_palette("tab10", len(unique_labels))
    base_color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    
    # Compute cosine distance from each data point to its corresponding NLP anchor.
    cos_dists = []
    for idx, row in df.iterrows():
        lbl = row["label"]
        class_idx = int(lbl)
        anchor_coord = anchors_2d[class_idx]
        point_coord = np.array([row["x"], row["y"]])
        sim = cosine_similarity(point_coord, anchor_coord)
        cos_dists.append(1 - sim)
    df["cos_dist"] = cos_dists
    
    # --- Aggregate: compute average cosine distances for intra- and inter-class for each class.
    records = []
    for lbl in unique_labels:
        class_idx = int(lbl)
        anchor = anchors_2d[class_idx]
        pts_intra = df[df["label"] == lbl][["x", "y"]].values
        if len(pts_intra) > 0:
            intra_cos = np.mean([1 - cosine_similarity(pt, anchor) for pt in pts_intra])
        else:
            intra_cos = np.nan
        pts_inter = df[df["label"] != lbl][["x", "y"]].values
        if len(pts_inter) > 0:
            inter_cos = np.mean([1 - cosine_similarity(pt, anchor) for pt in pts_inter])
        else:
            inter_cos = np.nan
        records.append({"Class": lbl, "Type": "Intra Cosine", "Cosine Distance": intra_cos})
        records.append({"Class": lbl, "Type": "Inter Cosine", "Cosine Distance": inter_cos})
    
    bar_df = pd.DataFrame(records)
    bar_df["Class"] = bar_df["Class"].astype(int)
    bar_df = bar_df.sort_values("Class")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bar_df, x="Class", y="Cosine Distance", hue="Type", palette="viridis")
    plt.xlabel("Class")
    plt.ylabel("Average Cosine Distance")
    plt.title("Average Cosine Distance to NLP Anchor per Class\n(Intra vs. Inter)")
    plt.tight_layout()
    plt.savefig("bar_plot_cosine_distance_from_saved.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir_path = "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad"
    graph_tsne(checkpoint_dir_path)
    bar_plot_cosine_distances_from_saved()



