#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=test_igpt
#SBATCH --output=test_igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/minGPT/test.py '/scratch/eo41/brady_1/test/novel_unseen' --batch_size 2 --data_cache '/scratch/eo41/minGPT/data_model_cache/brady_1_test_novel_unseen.pth' --model_cache '/scratch/eo41/minGPT/data_model_cache/model_24l_8h_512e_1b_64d_0.05lr_SGDop_3ep_4seed_brady_1_study.pt' --d_img 64 --save_name 'model_24l_8h_512e_1b_64d_0.05lr_SGDop_3ep_4seed_brady_1_test_novel_unseen'

echo "Done"
