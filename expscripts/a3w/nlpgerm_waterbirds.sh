# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_waterbirds_subpop_v2/\
#     --command_launcher multi_gpu\
#     --algorithms NLPGERM    \
#     --datasets WILDSWaterbirds  \
#     --n_hparams_from 0  \
#     --n_hparams 2       \
#     --n_trials 3       \
#     --steps 5000      \
#     --test_envs 4 5        \
#     --holdout_fraction 0.2 \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --skip_confirmation     \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_waterbirds_subpop_v2/\
#     --command_launcher multi_gpu\
#     --algorithms NLPGERM    \
#     --datasets WILDSWaterbirds  \
#     --n_hparams_from 2  \
#     --n_hparams 2       \
#     --n_trials 3       \
#     --steps 5000      \
#     --test_envs 4 5        \
#     --holdout_fraction 0.2 \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --skip_confirmation

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_waterbirds_subpop_class_balanced/    \
    --command_launcher multi_gpu\
    --algorithms NLPGERM    \
    --datasets WILDSWaterbirds  \
    --n_hparams_from 1  \
    --n_hparams 3       \
    --n_trials 3       \
    --steps 5000      \
    --test_envs 4 5        \
    --holdout_fraction 0.2 \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "class_balanced": true}' \
    --skip_confirmation

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_waterbirds_subpop_v2/\
#     --command_launcher multi_gpu\
#     --algorithms NLPGERM    \
#     --datasets WILDSWaterbirds  \
#     --n_hparams_from 10  \
#     --n_hparams 15       \
#     --n_trials 3       \
#     --steps 5000      \
#     --test_envs 4 5        \
#     --holdout_fraction 0.2 \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --skip_confirmation     \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_waterbirds_subpop_v2/\
#     --command_launcher multi_gpu\
#     --algorithms NLPGERM    \
#     --datasets WILDSWaterbirds  \
#     --n_hparams_from 15  \
#     --n_hparams 20       \
#     --n_trials 3       \
#     --steps 5000      \
#     --test_envs 4 5        \
#     --holdout_fraction 0.2 \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --skip_confirmation