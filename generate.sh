#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=gen
#SBATCH --output=gen_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/generate.py --condition 'half' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_A_2.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/model_A_2.pt'

echo "Done"
