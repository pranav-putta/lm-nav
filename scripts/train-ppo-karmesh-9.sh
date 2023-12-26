#!/bin/bash
#SBATCH --job-name train-ppo 
#SBATCH --output logs/train-ppo-9.out
#SBATCH --error logs/train-ppo-9.err
#SBATCH --gpus a40:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --exclude "xaea-12,nestor,heistotron,gundam,baymax,chappie,tachikoma"
#SBATCH --ntasks-per-node 4
#SBATCH --partition cvmlp-lab
#SBATCH --qos short
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

zsh
source ~/.zshrc
mamba activate lmnav

torchrun --standalone --nnodes=1 --nproc_per_node=4 lmnav/ppo_train.py train/nav_llama/1env_karmesh/rl/lora+clip+karmesh+9

# description: normalized advantages, increased clip range, increased lr over experiment 3, long warmup critic over experiment 4
