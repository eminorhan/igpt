#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda/11.1.74

#python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/A_1fps_288s' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_A_rcrop.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 28 --resume '/scratch/eo41/minGPT/data_model_cache/model_12l_8h_512e_32b_rcrop_A.pt'

#python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 32 --resume '' --subject 'SAY'

python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 16 --n_layer 24 --n_head 8 --n_emb 512 --resume '' --subject 'SAY'

echo "Done"
