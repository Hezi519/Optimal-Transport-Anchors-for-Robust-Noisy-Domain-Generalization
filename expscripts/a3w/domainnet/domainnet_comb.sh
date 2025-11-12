python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/erm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms ERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization \

# ---- irm ----
python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/irm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms IRM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization    \

# ---- groupdrop ----
python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/groupdro_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms GroupDRO     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization

# ----mixup----
python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/mixup_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms Mixup     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization \ 

# ----vrex -----
python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/vrex_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms VREx     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization    \

# ---- a3w ----
python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 0    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 1    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 2    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 3    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 4    \
    --task domain_generalization    \

python -m domainbed.scripts.sweep relaunch\
    --data_dir=/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data \
    --output_dir=./results/nlpgerm_dn_noise0.1/  \
    --command_launcher multi_gpu    \
    --algorithms NLPGERM     \
    --datasets DomainNet  \
    --skip_confirmation \
    --n_hparams_from 0  \
    --n_hparams 2       \
    --n_trials 2       \
    --holdout_fraction 0.2  \
    --hparams '{"flip_prob":0.1, "study_noise":1, "mapsty":"itera", "lambda":1.0, "temp":10, "iter_freq":10}' \
    --steps 5000      \
    --test_envs 5    \
    --task domain_generalization