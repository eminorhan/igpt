#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=igpt
#SBATCH --output=igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/evaluate_model.py '/scratch/eo41/minGPT/saycam_labeled' --traindata_cache '/scratch/eo41/minGPT/data_model_cache/data_A.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/model_A.pt'

echo "Done"
