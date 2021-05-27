#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
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

# srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 3 --n_layer 24 --n_head 8 --n_emb 512 --resume '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_3b_SAY.pt' --subject 'SAY'

# srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/quinn/train/dogs' --data_cache '/scratch/eo41/minGPT/data_model_cache/quinn_dogs.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 3 --n_layer 24 --n_head 8 --n_emb 512 --resume '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_3b_SAY.pt' --finetune --subject 'SAY'

# srun python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/quinn/val/dogs' --data_cache '/scratch/eo41/minGPT/data_model_cache/quin_dogs_val.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 2 --n_layer 24 --n_head 8 --n_emb 512 --resume '/scratch/eo41/minGPT/data_model_cache/model_quinn_cat.pt' --finetune --subject 'SAY' --epochs 1

echo "Done"
