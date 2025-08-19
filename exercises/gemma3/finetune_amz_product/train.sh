#!/bin/bash -l

#SBATCH --job-name=vlm
#SBATCH --time=999:000
#SBATCH --nodelist=hpe159,hpe160
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="gemma"
#SBATCH --output=gemma_ft_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v4.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/gemma3/finetune_amz_product'


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_addr=${nodes[0]}  # 첫 번째 노드를 rendezvous endpoint로 사용 # master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export NCCL_SOCKET_IFNAME=eno
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO # DETAIL
export TORCH_CPP_LOG_LEVEL=INFO 

export HF_TOKEN=''
export WANDB_KEY=''

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    torchrun --nproc_per_node=8 --nnodes=2 --rdzv_id 2523525 --master_addr $master_node --master_port 8882 --rdzv_endpoint $master_addr:$master_port train.py --hf_token $HF_TOKEN \
                                        --wandb_key $WANDB_KEY \
                                        --do_train \
                                        --logging_strategy steps \
                                        --logging_steps 1 \
                                        --per_device_train_batch_size 4


# 1 epoch 학습. 데이터수는 1345개
# 배치사이즈 8은 OOM
# 배치사이즈 1 + GPU A100 4개 -> 9분 50초
# 배치사이즈 4 + GPU A100 4개 -> 7분 40초
