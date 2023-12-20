#!/bin/bash
#SBATCH --job-name train-ppo 
#SBATCH --output logs/train-ppo-3.out
#SBATCH --error logs/train-ppo-3.err
#SBATCH --gpus a40:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --exclude "xaea-12,nestor,heistotron,gundam,baymax"
#SBATCH --ntasks-per-node 4
#SBATCH --partition cvmlp-lab
#SBATCH --qos short
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

zsh
source ~/.zshrc
mamba activate lmnav

torchrun --standalone --nnodes=1 --nproc_per_node=4 lmnav/ppo_train.py train/nav_llama/1env_karmesh/rl/lora+clip+karmesh+4

# description: normalized advantages, increased clip range, increased lr over experiment 3
