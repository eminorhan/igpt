#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=generate_igpt
#SBATCH --output=generate_igpt_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/minGPT/generate.py --condition 'half' --n_samples 4 --data_cache '/scratch/eo41/minGPT/data_model_cache/imagenet_train.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/imagenet_pretrained/model_12_24l_8h_512e_32b_64d_0.0005lr_Adamop_100ep_0seed_imagenet.pt'

echo "Done"
