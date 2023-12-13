#!/bin/bash
#SBATCH --job-name dgenpost-1ek 
#SBATCH --output logs/dgenpost-1ek.out
#SBATCH --error logs/dgenpost-1ek.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --exclude "xaea-12"
#SBATCH --ntasks-per-node 8
#SBATCH --partition cvmlp-lab
#SBATCH --qos short
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

zsh
source ~/.zshrc
mamba activate lmnav

python lmnav/dataset/offline_trajectory_dataset_post.py datagen/imagenav_karmesh_data_gen_1env
