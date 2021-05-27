#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda/11.1.74

#python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 32 --resume '' --subject 'SAY'

#python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/saycam/SAY_half_fps' --data_cache '/scratch/eo41/minGPT/data_model_cache/data_SAY_half_fps.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 32 --n_layer 12 --n_head 8 --n_emb 512 --resume '' --subject 'SAY'

python -u /scratch/eo41/minGPT/train.py '/scratch/eo41/minGPT/bomba/exemplars/s' --data_cache '/scratch/eo41/minGPT/data_model_cache/bomba_s.pth' --save_dir '/scratch/eo41/minGPT/data_model_cache' --batch_size 3 --n_layer 24 --n_head 8 --n_emb 512 --resume '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_3b_SAY.pt' --finetune --subject 'SAY'

echo "Done"
