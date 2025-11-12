python -m domainbed.scripts.sweep launch\
    --data_dir=./data/ \
    --output_dir=./results/noise_waterbirds/\
    --command_launcher multi_gpu\
    --algorithms ERM   \
    --datasets WILDSWaterbirds  \
    --n_hparams 1       \
    --n_trials 1       \
    --steps 5000      \
    --test_envs 4 5        \
    --holdout_fraction 0 \
    --hparams '{"flip_prob":0.25, "study_noise":1}'
    # --hparams '{"flip_prob":0.1, "study_noise":1}' \