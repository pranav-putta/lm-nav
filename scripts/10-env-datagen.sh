#!/bin/bash
#SBATCH --job-name 10e-datagen
#SBATCH --output logs/10e-datagen.out
#SBATCH --error logs/10e-datagen.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --ntasks-per-node 8
#SBATCH --partition long
#SBATCH --signal USR1@600

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

mamba activate lmnav

python lmnav/dataset/offline_data_gen.py datagen/imagenav_data_gen_10envs

