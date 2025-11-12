# No noise
python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \


python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \


python -m domainbed.scripts.sweep launch\
    --data_dir=./data \
    --output_dir=./results_na/pacs_baseline/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets PACS  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 5       \
    --n_trials 3       \
    --holdout_fraction 0.2  \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization