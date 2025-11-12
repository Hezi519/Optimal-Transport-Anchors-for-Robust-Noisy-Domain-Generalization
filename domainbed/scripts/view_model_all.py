import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- DomainBed imports ---
from domainbed import algorithms, datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc

def build_eval_loaders(args, model_hparams):
    """
    Rebuild the dataset and evaluation loaders as in your original snippet.
    Returns:
      evals: a list of (name, loader, weights) 
      dataset: the constructed DomainBed dataset object
    """
    data_name = args['dataset'].split('&')[0]
    # dataset_class = getattr(datasets, data_name)
    # dataset = dataset_class(args['data_dir'], args['test_envs'], model_hparams)
    dataset = vars(datasets)[data_name](args['data_dir'],
    args['test_envs'], model_hparams)

    # Example heuristic for GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[1] / 1e9
        if free_mem > 20:
            print("Detected large GPU memory. Setting num_workers=2")
            dataset.N_WORKERS = 1

    # Build in_splits/out_splits
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env,
            int(len(env)*args['holdout_fraction']),
            misc.seed_hash(args['trial_seed'], env_i)
        )
        if env_i in args['test_envs']:
            uda, in_ = misc.split_dataset(
                in_,
                int(len(in_)*args['uda_holdout_fraction']),
                misc.seed_hash(args['trial_seed'], env_i)
            )
        in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        if len(out) > 0:
            out_splits.append((out, out_weights))

    # We'll evaluate on test_in_splits + out_splits
    test_in_splits = [in_splits[i] for i in range(len(in_splits)) if i in args['test_envs']]
    eval_loaders = [
        FastDataLoader(
            dataset=env,
            batch_size=model_hparams['test_batch_size'],
            num_workers=dataset.N_WORKERS
        )
        for env, _ in (test_in_splits + out_splits)
    ]

    eval_weights = [None for _, weights in (test_in_splits + out_splits)]
    eval_loader_names = [
        f"env{i}_in" for i in range(len(in_splits)) if i in args['test_envs']
    ]
    eval_loader_names += [f"env{i}_out" for i in range(len(out_splits))]

    # Make a list of (name, loader, weights)
    evals = list(zip(eval_loader_names, eval_loaders, eval_weights))
    return evals, dataset


def get_features_for_model(evals, model):
    """
    Given a list of evals (each is (name, loader, _)) and a PyTorch model (with a .featurizer),
    loop over each loader and collect its features and labels in order.
    Returns:
      all_features, all_labels (both Torch tensors on CPU).
    """
    model.eval()
    device = next(model.parameters()).device

    all_features = []
    all_labels = []

    with torch.no_grad():
        for name, loader, _ in evals:
            for x, y in loader:
                x = x.to(device)
                feats = model.featurizer(x)  # shape [batch_size, feat_dim]
                all_features.append(feats.cpu())
                all_labels.append(y)  # stays on CPU by default

    all_features = torch.cat(all_features, dim=0)  # shape [N, feat_dim]
    all_labels = torch.cat(all_labels, dim=0)      # shape [N]
    return all_features, all_labels


def ensemble_tsne(model_dirs):
    """
    1) Load info from the first checkpoint (to build dataset/hparams).
    2) Build eval loaders with original DomainBed approach.
    3) For each model, load checkpoint and gather features → store in a list.
    4) Average these features across models → ensemble_features.
    5) Run t-SNE → single 2D plot.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the first checkpoint to get shape, hparams, etc.
    first_ckpt = torch.load(
        os.path.join(model_dirs[0], "model_best.pkl"), 
        map_location=device
    )
    input_shape = first_ckpt["model_input_shape"]
    num_classes = first_ckpt["model_num_classes"]
    num_domains = first_ckpt["model_num_domains"]
    model_hparams = first_ckpt["model_hparams"]
    args = first_ckpt["args"]

    # 1) Build evaluation loaders using original snippet logic
    evals, dataset = build_eval_loaders(args, model_hparams)

    # We'll store each model's entire feature array in all_model_features.
    # Then we'll average across models at the end.
    all_model_features = []
    all_labels = None

    # 2) Loop over each directory, load the model, gather features
    for i, folder in enumerate(model_dirs):
        ckpt_path = os.path.join(folder, "model_best.pkl")
        print(f"Loading model from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Re-instantiate the model (like your NLPGERM class or whichever)
        model = algorithms.NLPGERM(
            input_shape,
            num_classes,
            num_domains,
            model_hparams
        )

        # Optional: remove "maplayers" from state_dict
        raw_sd = checkpoint["model_dict"]
        filtered_sd = {
            k: v for k, v in raw_sd.items() if "maplayers" not in k
        }

        model.load_state_dict(filtered_sd, strict=False)
        model.to(device)

        # 3) Collect features for this model
        feats, labels = get_features_for_model(evals, model)
        if i == 0:
            # record labels from first model pass
            all_labels = labels.numpy()

        all_model_features.append(feats.numpy())

    # 4) Stack & average
    # shape: [num_models, N, feat_dim]
    stacked = np.stack(all_model_features, axis=0)
    # shape: [N, feat_dim]
    ensemble_features = stacked.mean(axis=0)

    # 5) Run t-SNE on the averaged features
    print("Running t-SNE on ensemble features...")
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(ensemble_features)

    # 6) Plot
    plt.figure(figsize=(8,8))
    plt.scatter(features_2d[:,0], features_2d[:,1],
                c=all_labels, cmap="jet", alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE on Ensemble-Averaged Featurizer Outputs")
    plt.savefig("ensemble_tsne_plot.png")
    plt.show()


if __name__ == "__main__":
    # Gather all experiment folders with 'model_best.pkl'
    model_dirs = []
    base_dir = "results/nlpgerm_pacs_ma/"
    dirs = os.listdir(base_dir)
    for folder in dirs:
        if os.path.exists(os.path.join(base_dir, folder, "model_best.pkl")):
            model_dirs.append(os.path.join(base_dir, folder))

    print(f"Found {len(model_dirs)} model directories.")
    ensemble_tsne(model_dirs)
