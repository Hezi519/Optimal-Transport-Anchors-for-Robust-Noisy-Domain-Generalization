python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization   \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0.2/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.2, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0.3/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization   \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0.4/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.4, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization   \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_pacs_noise_level_analysis_v2/noise0.5/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.5, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --seed 1   \
    --task domain_generalization