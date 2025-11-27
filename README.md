The algorithm.py have both weak OT and EOT version of align_loss, please remember to choose the one you need and comment out the other one before running.

Use python -m domainbed.scripts.train   --data_dir /home/u/data   --output_dir ./results/choose_your_own   --algorithm OT 
  --dataset PACS   --test_env 0   --steps 10000   --holdout_fraction 0.2   --hparams '{
    "flip_prob": 0.05,
    "study_noise": 1,
    "mapsty": "fixed",
    "lambda": 1.0,
    "resnet18": 1,
    "batch_size": 16,
    "resnet_pretrained": true,
    "temp": 1.0,
    "ot_reg": 0.1,
    "lr": 1e-4
  }'
to run our OT methods. 

Use python -m domainbed.scripts.train   --data_dir /home/u/data   --output_dir ./results/choose_your_own   --algorithm ERM 
  --dataset PACS   --test_env 0   --steps 10000   --holdout_fraction 0.2   --hparams '{
    "flip_prob": 0.05,
    "study_noise": 1,
    "mapsty": "fixed",
    "lambda": 1.0,
    "resnet18": 1,
    "batch_size": 16,
    "resnet_pretrained": true,
    "temp": 1.0,
    "ot_reg": 0.1,
    "lr": 1e-4
  }'
