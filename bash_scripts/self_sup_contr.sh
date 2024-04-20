#!/bin/bash
#SBATCH -A dep # account
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/ps2g02/Graph-Anomaly-Detection # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o error_folder/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e error_folder/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Para pedir gráficas

python3 /hhome/ps2g02/Graph-Anomaly-Detection/train_contrastive_selfsupervisedV2.py $1