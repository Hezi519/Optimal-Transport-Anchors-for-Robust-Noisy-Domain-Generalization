python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_ablation/seed1/noweight/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoWeight     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "iter_freq":10, "temp":10}' \
    --seed 1    \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_ablation/seed1/nonlp/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM_NoNLP     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "iter_freq":10, "temp":10}' \
    --seed 1  \
    --steps 5000      \
    --test_envs 0   \
    --task domain_generalization    \

python -m domainbed.scripts.sweep launch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_ablation/seed1/baseline/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.25, "study_noise":1, "mapsty":"itera", "lambda":1.0, "iter_freq":10, "temp":10}' \
    --seed 1  \
    --steps 5000      \
    --test_envs 0   \
    --task domain_generalization



