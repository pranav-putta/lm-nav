#!/bin/bash
#SBATCH --job-name dgen-1ek 
#SBATCH --output logs/dgen-1ek.out
#SBATCH --error logs/dgen-1ek.err
#SBATCH --gpus a40:8
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

python lmnav/dataset/offline_data_gen.py datagen/imagenav_karmesh_data_gen_1env
