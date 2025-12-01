# Optimal Transport Anchors for Robust-Noisy Domain Generalization

This repository is based on the implementation of [Understanding Domain Generalization: A Noise Robustness Perspective](https://github.com/qiaoruiyt/NoiseRobustDG) in ICLR 2024 by [Rui Qiao](https://qiaoruiyt.github.io) and [Bryan Kian Hsiang Low](https://www.comp.nus.edu.sg/~lowkh/research.html).

## To replicate our results

Like Domainbed, this repo can be easily setup without installing many other packages if you have already setup a Python environment with the latest PyTorch. The required packages can be installed by:
```sh
pip install -r domainbed/requirements.txt
```

Download the datasets (The gdown version can affect the downloading of certain datasets from Google Drive. Please consider using the recommended version in `requirements.txt`):

```sh
python3 -m domainbed.scripts.download \
       --data_dir=~/data
```

Train alignments with the exact configuration as ours:

```sh
source run.sh
```

## To reproduce our exact numbers

We have provided the data logs in our training runs. To reproduce our exact numbers, run:

```sh
python -m domainbed.scripts.extract_results
```