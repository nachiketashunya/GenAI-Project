#!/bin/bash
#SBATCH --job-name=diffopt_run
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 ##Define number of GPUs
#SBATCH --output=logs_e40/diffopt_%j.log

echo "JOb Submitted"

module load anaconda3/2024
conda init
source ~/.bashrc
conda activate meesho

python /iitjhome/m23csa016/meesho_code/training_code/clipvit_uf_hyper_tune_val_other.py
# python nosplit.py

