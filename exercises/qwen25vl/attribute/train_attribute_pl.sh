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
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/qwen25vl/attribute'

export HF_TOKEN=''
export WANDB_KEY=''

# ---------------- 실행 ----------------
srun --gpus-per-task=1 \
     --container-image "$CONTAINER_IMAGE_PATH" \
     --container-mounts /purestorage:/purestorage,"$CACHE_FOR_PATH":/home/$USER/.cache \
     --no-container-mount-home \
     --container-writable \
     --container-workdir "$MY_WORKSPACE_PATH" \
     python train_attribute_pl.py --nnodes $SLURM_NNODES --ngpus $SLURM_NTASKS_PER_NODE --hf_token $HF_TOKEN --wandb_key $WANDB_KEY --pl_strategy auto