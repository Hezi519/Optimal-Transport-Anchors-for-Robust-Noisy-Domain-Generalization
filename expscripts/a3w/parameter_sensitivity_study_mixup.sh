# # lambda
# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lambda/0.1  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":0.1}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lambda/0.5  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":0.5}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lambda/1.0  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \


# # lr
# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lr/1e-2/  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "lr": 1e-2}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lr/1e-3/  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "lr": 1e-3}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lr/1e-4/ \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "lr": 1e-4}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lr/1e-5/  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "lr": 1e-5}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/lr/1e-6/ \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "lr": 1e-6}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization


# # noise
# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=./data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0/  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0, "study_noise":1, "mapsty":"itera", "lambda":1.0'} \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=./data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0.1/  \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=./data \
#     --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0.2/ \
#     --command_launcher multi_gpu    \
#     --algorithms Mixup     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 1       \
#     --n_trials 1       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.2, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --seed 1    \
#     --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=./data \
    --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0.3/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 1000      \
    --test_envs 0    \
    --seed 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0.4/ \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.4, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 1000      \
    --test_envs 0    \
    --seed 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results/parameter_sensitivity_analysis_mixup/noise/0.5/ \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 1       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.5, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 1000      \
    --test_envs 0    \
    --seed 1    \
    --task domain_generalization