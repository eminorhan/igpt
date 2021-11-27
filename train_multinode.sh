#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=300GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=train_igpt
#SBATCH --output=train_igpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=8

module purge
module load cuda/11.1.74

#srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/S_clean_labeled_data_1fps_5' --data_cache '/scratch/eo41/minGPT/data_model_cache/labeled_S.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 1 --n_layer 24 --n_head 8 --n_emb 512 --resume '' --subject 'labeled_S'

srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/brady/all' --data_cache '/scratch/eo41/minGPT/data_model_cache/brady_all.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --epochs 1 --batch_size 1 --n_layer 24 --n_head 8 --n_emb 512 --resume '' --subject 'brady'

echo "Done"
