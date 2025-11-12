python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./result_a3w/irm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization