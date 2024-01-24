#!/bin/bash
#SBATCH --job-name train-1ek 
#SBATCH --output logs/train-1ek+lora16.out
#SBATCH --error logs/train-1ek+lora16.err
#SBATCH --gpus a40:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --exclude "xaea-12"
#SBATCH --ntasks-per-node 4
#SBATCH --partition cvmlp-lab
#SBATCH --qos short
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

zsh
source ~/.zshrc
mamba activate lmnav

torchrun --standalone --nnodes=1 --nproc_per_node=4 lmnav/bc_train.py train/nav_llama/1env_karmesh/bc/lora16+clip+karmesh
