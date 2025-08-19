set -euo pipefail        # 안전한 스크립트 옵션
# -------------------------
# 1) 환경 변수 / 경로 설정
# -------------------------
export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/qwen_4.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLM'

# -------------------------
# 2) 학습 관련 변수
# -------------------------
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
DATASETS="object365"
OUTPUT_DIR="./output_qwen_finetune"
CACHE_DIR="/purestorage/AILAB/AI_1/tyk/0_Software/cache"
DEEPSPEED_CFG="/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/qwen25vl/finetune_detection/scripts/zero3.json"


MASTER_ADDR="127.0.0.1"
MASTER_PORT="$(shuf -i 20000-29999 -n 1)"

# 해상도·픽셀 관련 숫자(미리 계산)
MAX_PIXELS=$((576*28*28))
MIN_PIXELS=$((16*28*28))
VIDEO_MAX_FRAME_PIXELS=$((1664*28*28))
VIDEO_MIN_FRAME_PIXELS=$((256*28*28))

# -------------------------
# 3) torchrun 인자를 배열로 정의
# -------------------------
args=(
  # Core
  --model_name_or_path "$MODEL_PATH"
  --tune_mm_llm True
  --tune_mm_vision False
  --tune_mm_mlp False
  --dataset_use "$DATASETS"
  --output_dir "$OUTPUT_DIR"
  --cache_dir "$CACHE_DIR"

  # Precision & Memory
  --bf16
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 4

  # Learning rate
  --learning_rate 2e-7
  --mm_projector_lr 1e-5
  --vision_tower_lr 1e-6
  --optim adamw_torch

  # Sequence / Packing
  --model_max_length 4096
  --data_flatten True
  --data_packing True

  # Image
  --max_pixels "$MAX_PIXELS"
  --min_pixels "$MIN_PIXELS"

  # Video
  --base_interval 2
  --video_max_frames 8
  --video_min_frames 4
  --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS"
  --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS"

  # Training schedule
  --num_train_epochs 3
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --weight_decay 0.01

  # Logging / checkpoints
  --logging_steps 10
  --save_steps 500
  --save_total_limit 3

  # Advanced
  # --deepspeed "$DEEPSPEED_CFG"
)

# -------------------------
# 4) 학습 실행
# -------------------------
torchrun \
  --nproc_per_node=1 \
  --master_port="$MASTER_PORT" \
  qwenvl/train_qwen.py \
  "${args[@]}"



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