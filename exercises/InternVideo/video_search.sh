#!/bin/bash -l

#SBATCH --job-name=vlm
#SBATCH --time=999:000
##SBATCH --partition=80g
#SBATCH --nodelist=nv180
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="cc_train_pl"
#SBATCH --output=./logs/cc_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v10.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/LMM/InternVideo'


# ---------------- 실행 ----------------
srun --container-image "$CONTAINER_IMAGE_PATH" \
     --container-mounts /purestorage:/purestorage,"$CACHE_FOR_PATH":/home/$USER/.cache \
     --no-container-mount-home \
     --container-writable \
     --container-workdir "$MY_WORKSPACE_PATH" \
     python video_search.py