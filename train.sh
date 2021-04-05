#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_A.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --resume '/scratch/eo41/minGPT/data_model_cache/model_12l_8h_512e_32b_A.pt'

echo "Done"
