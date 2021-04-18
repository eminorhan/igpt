#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/frames_for_half_train' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_A.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 6 --resume ''

echo "Done"
