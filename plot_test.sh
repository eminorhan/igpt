#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=plot
#SBATCH --output=plot_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/plot.py

echo "Done"
