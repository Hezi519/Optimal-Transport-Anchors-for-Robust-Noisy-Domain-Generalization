# ---- noise 5% ----
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.05/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.05, "study_noise":1}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.05/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.05, "study_noise":1}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.05/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.05, "study_noise":1}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.05/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.05, "study_noise":1}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

# ---- noise 10% ----
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

# ---- noise 15% ----
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.15/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.15, "study_noise":1}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.15/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.15, "study_noise":1}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.15/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.15, "study_noise":1}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.15/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.15, "study_noise":1}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \


# ---- noise 20% ----
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.2/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.2, "study_noise":1}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.2/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.2, "study_noise":1}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.2/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.2, "study_noise":1}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.2/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.2, "study_noise":1}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

# ---- noise 30% ----
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.3/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.3/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.3/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/noise0.3/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.3, "study_noise":1}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization