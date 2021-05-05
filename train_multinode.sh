#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

module purge
module load cuda/11.1.74

srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 8 --n_layer 12 --n_head 8 --n_emb 512 --resume '' --subject 'SAY'

echo "Done"
