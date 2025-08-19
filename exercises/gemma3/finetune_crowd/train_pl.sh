#!/bin/bash -l

#SBATCH --job-name=vlm
#SBATCH --time=999:000
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="gemma"
#SBATCH --output=gemma_ft_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v12.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/gemma3/finetune_crowd'

export HF_TOKEN=''
export WANDB_KEY=''

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python train_pl.py --nnodes $SLURM_NNODES --ngpus $SLURM_NTASKS_PER_NODE

