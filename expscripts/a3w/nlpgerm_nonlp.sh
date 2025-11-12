python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_vlcs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets VLCS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_pacs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_pacs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_pacs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_pacs/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_oh/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \


python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_oh/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \


python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_oh/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \


python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_oh/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets OfficeHome  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_terra  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets TerraIncognita  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_terra  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets TerraIncognita  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_terra  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets TerraIncognita  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_nonlp_terra  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets TerraIncognita  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 1       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization
