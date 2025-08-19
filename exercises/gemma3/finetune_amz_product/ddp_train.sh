#!/bin/bash -l
#SBATCH --job-name=vlm
#SBATCH --time=999:00              # 999시간이라면 999:00:00 이 맞습니다
#SBATCH --nodes=2
#SBATCH --nodelist=cubox12,cubox13
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="gemma"
#SBATCH --output=gemma_ft_%j.out

# ---------------- 경로 변수 ----------------
export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v4.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/gemma3/finetune_amz_product'

# ---------------- 런데뷰 주소 ----------------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=8882               # 사용 가능한 포트 하나 지정

# ---------------- 환경 변수 ----------------
# export NCCL_SOCKET_IFNAME=eno
export NCCL_SOCKET_IFNAME=eno
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO

export HF_TOKEN=''
export WANDB_KEY=''

# ---------------- 실행 ----------------
srun --container-image "$CONTAINER_IMAGE_PATH" \
     --container-mounts /purestorage:/purestorage,"$CACHE_FOR_PATH":/home/$USER/.cache \
     --no-container-mount-home \
     --container-writable \
     --container-workdir "$MY_WORKSPACE_PATH" \
     torchrun \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --nproc_per_node=8 \
       --node_rank="$SLURM_NODEID" \
       --rdzv_backend=c10d \
       --rdzv_id="$SLURM_JOB_ID" \
       --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
       train.py \
         --hf_token "$HF_TOKEN" \
         --wandb_key "$WANDB_KEY" \
         --do_train \
         --logging_strategy steps \
         --logging_steps 1 \
         --per_device_train_batch_size 4