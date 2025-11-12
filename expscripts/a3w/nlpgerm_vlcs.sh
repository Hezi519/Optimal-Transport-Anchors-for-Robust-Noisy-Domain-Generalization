python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization    \


python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization    \


python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \


python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_vlcs_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 3    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 3    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_vlcs_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets VLCS  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 3    \
#     --task domain_generalization

# -------------------------OH----------------------
python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_oh_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 3       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 3  \
#     --n_hparams 7       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep launch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 0    \
#     --task domain_generalization


python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_oh_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 1    \
#     --task domain_generalization


python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_oh_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 5  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 10  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization    \

# python -m domainbed.scripts.sweep relaunch\
#     --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
#     --output_dir=./results/nlpgerm_oh_ma/  \
#     --command_launcher multi_gpu    \
#     --algorithms NLPGERM     \
#     --datasets OfficeHome  \
#     --skip_confirmation \
#     --n_hparams_from 15  \
#     --n_hparams 5       \
#     --n_trials 3       \
#     --holdout_fraction 0.2  \
#     --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
#     --steps 5000      \
#     --test_envs 2    \
#     --task domain_generalization


python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_oh_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 5  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization


