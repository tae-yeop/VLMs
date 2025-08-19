#!/bin/bash -l

#SBATCH --job-name=vlm
#SBATCH --time=999:000
#SBATCH -p 80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --comment="VLM 테스트"
#SBATCH --output=./logs/qwenvl_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/qwen_4.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLM'


MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1) 

DATASETS="object365" 

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    torchrun --nproc_per_node=1 \
            --master_port=$MASTER_PORT \
            qwenvl/train_qwen.py \
            # Core Arguments
            --model_name_or_path $MODEL_PATH \  # [ModelArguments] Model identifier
            --tune_mm_llm True \                # [TrainingArguments] Train LLM or not
            --tune_mm_vision False \            # [TrainingArguments] Train VIT or not
            --tune_mm_mlp False \               # [TrainingArguments] Train MLP or not
            --dataset_use $DATASETS \           # [DataArguments] Dataset specification
            --output_dir $OUTPUT_DIR \          # Output directory for checkpoints
            --cache_dir $CACHE_DIR \            # [TrainingArguments] Model cache location

            # Precision & Memory
            --bf16 \                            # Use bfloat16 precision (Ampere+ GPUs)
            --per_device_train_batch_size 4 \   # Batch size per GPU
            --gradient_accumulation_steps 4 \   # Effective batch size multiplier

            # Learning Rate Configuration
            --learning_rate 2e-7 \              # Base learning rate
            --mm_projector_lr 1e-5 \            # [TrainingArguments] Projector-specific LR
            --vision_tower_lr 1e-6 \            # [TrainingArguments] Vision encoder LR
            --optim adamw_torch \               # [TrainingArguments] Optimizer selection

            # Sequence Configuration
            --model_max_length 4096 \           # [TrainingArguments] Max sequence length
            --data_flatten True \               # [DataArguments] Concatenate batch sequences
            --data_packing True \               # [DataArguments] Using packing data

            # Image Processing
            --max_pixels 576\*28\*28 \               # [DataArguments] Max image pixels (H*W) for image
            --min_pixels 16\*28\*28 \                # [DataArguments] Min image pixels for image
            # Video Processing
            --base_interval 2 \                      # [DataArguments] Sampling time interval (seconds) between frames
            --video_max_frames 8 \                   # [DataArguments] Max frames per video
            --video_min_frames 4 \                   # [DataArguments] Min frames per video
            --video_max_frame_pixels 1664\*28\*28 \  # [DataArguments] Max pixels within a frame
            --video_min_frame_pixels 256\*28\*28 \   # [DataArguments] Min pixels within a frame
            
            # Training Schedule
            --num_train_epochs 3 \              # Total training epochs
            --warmup_ratio 0.03 \               # LR warmup proportion
            --lr_scheduler_type "cosine" \      # Learning rate schedule
            --weight_decay 0.01 \               # L2 regularization strength
            
            # Logging & Checkpoints
            --logging_steps 10 \               # Log metrics interval
            --save_steps 500 \                 # Checkpoint save interval
            --save_total_limit 3 \             # Max checkpoints to keep
            
            # Advanced Options
            --deepspeed zero3.json \           # DeepSpeed configuration




# srun --container-image $CONTAINER_IMAGE_PATH \
#     --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
#     --no-container-mount-home \
#     --container-writable \
#     --container-workdir $MY_WORKSPACE_PATH \
#     python qwenvl/data/__init__.py
# torchrun --nproc_per_node=1 \
#             --master_port=$MASTER_PORT \
#             qwenvl/train_qwen.py \
#             --model_name_or_path 'Qwen/Qwen-7B-Chat' \