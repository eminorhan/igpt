#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=1
#SBATCH --job-name=finetune_igpt
#SBATCH --output=finetune_igpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=16

module purge
module load cuda/11.3.1

LR=0.0005
OPTIMIZER='Adam'

srun python -u /scratch/eo41/minGPT/finetune.py '/scratch/eo41/brady_1/study' --data_cache '/scratch/eo41/minGPT/data_model_cache/brady_1_study_imagenet.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --epochs 100 --batch_size 1 --optimizer $OPTIMIZER --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 6 --n_head 2 --n_emb 512 --resume '/scratch/eo41/minGPT/data_model_cache/imagenet_pretrained/model_12_6l_2h_512e_32b_64d_0.0005lr_Adamop_100ep_0seed.pt' --pretrain_data 'imagenet'

echo "Done"
