python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 3  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 6  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 9  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 12  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 15  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 18  \
    --n_hparams 3       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

# ----- PACS -----

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 0  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --skip_confirmation \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --n_hparams_from 3  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --skip_confirmation \
#     --n_hparams_from 6  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --skip_confirmation \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --n_hparams_from 9  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --skip_confirmation \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --n_hparams_from 12  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --skip_confirmation \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --n_hparams_from 15  \
#     --n_hparams 3       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_pacs/  \
#     --skip_confirmation \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets PACS  \
#     --n_hparams_from 18  \
#     --n_hparams 2       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization