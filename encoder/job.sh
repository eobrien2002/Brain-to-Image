#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --tasks-per-node=1
#SBATCH --time=12:30:00
#SBATCH --output=%N-%j.out
#SBATCH --mem=125G
#SBATCH --account=def-yalda          

source ../v3/venv/bin/activate
python -m torch.distributed.launch --nproc_per_node=2 trainer_aligner.py

