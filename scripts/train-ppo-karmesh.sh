#!/bin/bash
#SBATCH --job-name train-ppo 
#SBATCH --output logs/train-ppo.out
#SBATCH --error logs/train-ppo.err
#SBATCH --gpus a40:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --exclude "xaea-12,nestor"
#SBATCH --ntasks-per-node 8
#SBATCH --partition cvmlp-lab
#SBATCH --qos short
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

zsh
source ~/.zshrc
mamba activate lmnav

torchrun --standalone --nnodes=1 --nproc_per_node=8 lmnav/ppo_train.py train/nav_llama/1env_karmesh/rl/lora+clip+karmesh
