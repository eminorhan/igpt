#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --job-name=train_igpt
#SBATCH --output=train_igpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.1.74

LR=0.05

srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/brady_1/study' --data_cache '/scratch/eo41/minGPT/data_model_cache/brady_1_study.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --epochs 1 --batch_size 1 --optimizer 'SGD' --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 24 --n_head 8 --n_emb 512 --resume '' --subject 'brady_1_study'

echo "Done"
