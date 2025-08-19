#!/bin/bash -l

#SBATCH --job-name=vlm-finetune
#SBATCH --time=999:000
#SBATCH -p 80g
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --comment='SMOVLM 파인튜닝'
#SBATCH --qos=normal

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_v2.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/smolVLM'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    torchrun --nproc_per_node=8 finetune.py