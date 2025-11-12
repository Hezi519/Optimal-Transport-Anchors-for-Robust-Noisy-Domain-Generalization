# AAAW
import os
import torch
import numpy as np
from domainbed import algorithms, datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc

# --- Load checkpoint and model settings ---
checkpoint_dir_path = "./results/nlpgerm_pacs_ma/f06d4e635429b3289d4958267e3fbbad"
checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
input_shape   = checkpoint["model_input_shape"]
num_classes   = checkpoint["model_num_classes"]
num_domains   = checkpoint["model_num_domains"]
model_hparams = checkpoint["model_hparams"]
args          = checkpoint["args"]

# # Disable noise for visualization if desired.
# if model_hparams.get("flip_prob") is not None:
#     model_hparams["flip_prob"] = 0
#     model_hparams["study_noise"] = 0

# --- Build model (mapping layers are stored in the checkpoint) ---
state_dict = checkpoint["model_dict"]
model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
model.load_state_dict(state_dict, strict=False)
model.eval()
featurizer = model.featurizer.to("cpu")

# --- Build dataset and extract mapped features ---
data_name = args["dataset"].split("&")[0]
dataset_class = vars(datasets)[data_name]
dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
if torch.cuda.is_available():
    free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
    if free_mem_gb > 20:
        dataset_obj.N_WORKERS = 0

# -------------------------------
# Build a set of file paths that were noised.
# The dataset initialization code updates `dataset_obj.noisy_datasets`
# with only those samples that had their labels flipped.
noisy_paths = set()
for noisy_ds in dataset_obj.noisy_datasets:
    for sample in noisy_ds.samples:
        # Each sample is a tuple: (file_path, label)
        noisy_paths.add(sample[0])
        
print("Total number of noised samples:", len(noisy_paths))

# -------------------------------
# Generate NLP anchors and mapping layers.
model.set_nlp_anchor(dataset_obj)
nlpanchors = model.nlpanchor  # shape: (num_classes, anchor_dim)
anchors_np = nlpanchors.detach().cpu().numpy()
print("NLP anchors shape:", anchors_np.shape)

# # -------------------------------
# # Extract mapped features, labels, and file paths.
# all_features_list = []
# all_labels_list = []
all_paths = []

# We iterate over each environment in the dataset.
# Note: The original feature extraction loop uses misc.split_dataset to get a training split.
# For simplicity, we iterate over the entire dataset here.
for env in dataset_obj.datasets:
    for idx in range(len(env)):
        img, label = env[idx]
        # Get file path for the sample (from ImageFolder, sample = (file_path, label))
        file_path = env.samples[idx][0]
        
        # # Expand dims so that the model expects a batch of 1.
        # x = img.unsqueeze(0)
        # with torch.no_grad():
        #     raw_feat = featurizer(x.to("cpu"))
        #     # Pass the raw feature through the mapping layer corresponding to the sample’s label.
        #     mapped_feat = model.maplayers[label](raw_feat)
        # all_features_list.append(mapped_feat.squeeze(0).cpu().numpy())
        # all_labels_list.append(label)
        all_paths.append(file_path)

# if not all_features_list:
#     print("No data available!")
#     exit()

# # Convert lists to numpy arrays.
# features_arr = np.stack(all_features_list)  # shape: (N, feat_dim)
# labels_arr = np.array(all_labels_list)        # shape: (N,)

features_path = "graph/all_features.npy"  # shape: (N, mapped_dim)
labels_path   = "graph/all_labels.npy"    # shape: (N,)
anchors_path  = "graph/anchors.npy"        # shape: (num_classes, anchor_dim)

features_arr = np.load(features_path)
labels_arr   = np.load(labels_path)
anchors_np   = np.load(anchors_path)


def cosine_similarity_np(a, b):
    # Compute element-wise dot product along rows
    dot_product = np.sum(a * b, axis=1)
    # Compute L2 norms for each row in both arrays
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    # Return cosine similarity for each pair
    return dot_product / (norm_a * norm_b)

# -------------------------------
# Compute Euclidean distance between each sample's mapped feature and its corresponding NLP anchor.
# For each sample, we select the anchor based on its label.
# distances = np.linalg.norm(features_arr - anchors_np[labels_arr], axis=1)
distances = cosine_similarity_np(features_arr, anchors_np[labels_arr])


# -------------------------------
# Sort samples by distance (ascending order: closest to anchor first)
sorted_indices = np.argsort(distances)

# Select the first 50% (i.e. samples with the smallest distances to their anchors)
num_subset = len(sorted_indices) // 2
subset_indices = sorted_indices[:num_subset]

# -------------------------------
# Count how many samples in the closest 50% are noised.
# We use the file path as a unique identifier.
num_noisy = sum(1 for idx in subset_indices if all_paths[idx] in noisy_paths)
print("Total number of noised samples:", len(noisy_paths))
print("Number of noisy samples in the closest 50%:", num_noisy)



# # ERM
# import os
# import torch
# import numpy as np
# from domainbed import algorithms, datasets
# from domainbed.lib.fast_data_loader import FastDataLoader
# from domainbed.lib import misc

# # --- Load checkpoint and model settings ---
# checkpoint_dir_path = "./results/mixup_pacs_baseline/0ed15c75b1c9a9cfd23c3bf841dc9044"
# checkpoint_path = os.path.join(checkpoint_dir_path, "model_best.pkl")
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# input_shape   = checkpoint["model_input_shape"]
# num_classes   = checkpoint["model_num_classes"]
# num_domains   = checkpoint["model_num_domains"]
# model_hparams = checkpoint["model_hparams"]
# args          = checkpoint["args"]

# # # Disable noise for visualization if desired.
# # if model_hparams.get("flip_prob") is not None:
# #     model_hparams["flip_prob"] = 0
# #     model_hparams["study_noise"] = 0

# # --- Build model (mapping layers are stored in the checkpoint) ---
# state_dict = checkpoint["model_dict"]
# model = algorithms.NLPGERM(input_shape, num_classes, num_domains, model_hparams)
# model.load_state_dict(state_dict, strict=False)
# model.eval()
# featurizer = model.featurizer.to("cpu")

# # --- Build dataset and extract mapped features ---
# data_name = args["dataset"].split("&")[0]
# dataset_class = vars(datasets)[data_name]
# dataset_obj = dataset_class(args["data_dir"], args["test_envs"], model_hparams)
# if torch.cuda.is_available():
#     free_mem_gb = torch.cuda.mem_get_info()[1] / 1e9
#     if free_mem_gb > 20:
#         dataset_obj.N_WORKERS = 0

# # -------------------------------
# # Build a set of file paths that were noised.
# # The dataset initialization code updates `dataset_obj.noisy_datasets`
# # with only those samples that had their labels flipped.
# noisy_paths = set()
# for noisy_ds in dataset_obj.noisy_datasets:
#     for sample in noisy_ds.samples:
#         # Each sample is a tuple: (file_path, label)
#         noisy_paths.add(sample[0])
        
# print("Total number of noised samples:", len(noisy_paths))


# # -------------------------------
# # Extract mapped features, labels, and file paths.
# all_features_list = []
# all_labels_list = []
# all_paths = []

# # We iterate over each environment in the dataset.
# # Note: The original feature extraction loop uses misc.split_dataset to get a training split.
# # For simplicity, we iterate over the entire dataset here.
# for env in dataset_obj.datasets:
#     for idx in range(len(env)):
#         img, label = env[idx]
#         # Get file path for the sample (from ImageFolder, sample = (file_path, label))
#         file_path = env.samples[idx][0]
        
#         # Expand dims so that the model expects a batch of 1.
#         x = img.unsqueeze(0)
#         with torch.no_grad():
#             raw_feat = featurizer(x.to("cpu"))
#         all_features_list.append(raw_feat.squeeze(0).cpu().numpy())
#         all_labels_list.append(label)
#         all_paths.append(file_path)

# if not all_features_list:
#     print("No data available!")
#     exit()

# # Convert lists to numpy arrays.
# features_arr = np.stack(all_features_list)  # shape: (N, feat_dim)
# labels_arr = np.array(all_labels_list)        # shape: (N,)


# # -------------------------------
# # Compute Euclidean distance between each sample's mapped feature and its corresponding NLP anchor.
# # For each sample, we select the anchor based on its label.
# distances = np.linalg.norm(features_arr - anchors_np[labels_arr], axis=1)

# # -------------------------------
# # Sort samples by distance (ascending order: closest to anchor first)
# sorted_indices = np.argsort(distances)

# # Select the first 50% (i.e. samples with the smallest distances to their anchors)
# num_subset = len(sorted_indices) // 2
# subset_indices = sorted_indices[:num_subset]

# # -------------------------------
# # Count how many samples in the closest 50% are noised.
# # We use the file path as a unique identifier.
# num_noisy = sum(1 for idx in subset_indices if all_paths[idx] in noisy_paths)
# print("Total number of noised samples:", len(noisy_paths))
# print("Number of noisy samples in the closest 50%:", num_noisy)
