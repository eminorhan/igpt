#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_igpt
#SBATCH --output=train_igpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=48

module purge
module load cuda/11.3.1

LR=0.0005
OPTIMIZER='Adam'

srun python -u /scratch/eo41/minGPT/train_say.py '/scratch/eo41/say_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/saycam_train.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --epochs 100 --start-epoch 0 --batch_size 1 --optimizer $OPTIMIZER --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 24 --n_head 8 --n_emb 512 --resume ''

echo "Done"
