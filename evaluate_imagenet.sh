#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=eval
#SBATCH --output=eval_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/evaluate_imagenet.py --train_data_path '/scratch/work/public/imagenet/train' --val_data_path '/scratch/eo41/robust_vision/imagenet/val' --traindata_cache '/scratch/eo41/minGPT/data_model_cache/imagenet_trainset_64.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_1b_64d_ImageNet.pt' --probe_layer 12 --batch_size 128

echo "Done"
