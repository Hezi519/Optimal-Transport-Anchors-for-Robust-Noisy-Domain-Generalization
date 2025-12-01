hparams0=$(cat <<'HP'
{
  "flip_prob": 0.05,
  "study_noise": 1,
  "mapsty": "fixed",
  "lambda": 1.0,
  "resnet18": 1,
  "batch_size": 16,
  "resnet_pretrained": true,
  "temp": 1.0,
  "ot_reg": 0.1,
  "ot_unbalanced": [1, 1],
  "lr": 1e-4
}
HP
)

python -m domainbed.scripts.sweep launch \
  --data_dir=data \
  --output_dir=./results/erm_cos_uot_wot \
  --command_launcher local \
  --algorithm OT OTWeak NLPGERM ERM \
  --dataset VLCS PACS \
  --test_env 0 \
  --steps 10000 \
  --holdout_fraction 0.2 \
  --n_hparams 1 \
  --n_trials 1 \
  --hparams "$hparams0"

hparams1=$(cat <<'HP'
{
  "flip_prob": 0.05,
  "study_noise": 1,
  "mapsty": "fixed",
  "lambda": 1.0,
  "resnet18": 1,
  "batch_size": 16,
  "resnet_pretrained": true,
  "temp": 1.0,
  "ot_reg": 0.1,
  "ot_unbalanced": null,
  "lr": 1e-4
}
HP
)

python -m domainbed.scripts.sweep launch \
  --data_dir=data \
  --output_dir=./results/eot \
  --command_launcher local \
  --algorithm OT \
  --dataset VLCS PACS \
  --test_env 0 \
  --steps 10000 \
  --holdout_fraction 0.2 \
  --n_hparams 1 \
  --n_trials 1 \
  --hparams "$hparams1"