#!/bin/bash
#SBATCH --job-name=dslab_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00  # Adjust time as needed
#SBATCH --gres=gpu:1  

export PYTHONPATH="${PYTHONPATH}:/itet-stor/elucas/net_scratch/generative_inversion/"

python /itet-stor/elucas/net_scratch/generative_inversion/evaluation/eval.py