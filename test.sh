#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=test_igpt
#SBATCH --output=test_igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/test.py '/scratch/eo41/brady/all' --batch_size 2 --data_cache '/scratch/eo41/minGPT/data_model_cache/brady_all.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_8b_64d_0.0005lr_1ep_brady.pt'

echo "Done"
