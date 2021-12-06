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

CONDITIONS=(novel_seen novel_unseen exemplar_seen exemplar_unseen state_seen state_unseen)

for CONDITION in "${CONDITIONS[@]}"
do
    python -u /scratch/eo41/minGPT/test.py /scratch/eo41/brady_1/test/$CONDITION \
        --batch_size 2 \
        --d_img 64 \
        --data_cache /scratch/eo41/minGPT/data_model_cache/brady_1_test_${CONDITION}.pth \
        --model_cache /scratch/eo41/minGPT/data_model_cache/brady_1_0/model_49_24l_8h_512e_16b_64d_0.0005lr_Adamop_100ep_0seed_brady_1_study.pt \
        --save_name model_49_24l_8h_512e_8b_64d_0.0005lr_Adamop_100ep_0seed_brady_1_test_$CONDITION
done
echo "Done"
