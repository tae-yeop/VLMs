from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class CustomArguments:
    # 기본값 없는걸 앞에 몰아서 써야함. 안그럼 에러
    # 이게 싫으면 모든 항목에 기본값을 deafualt=None을 주도록 함
    nnodes: int = field(metadata={"help": "Number of Nodes"})
    ngpus: int = field(metadata={"help": "Number of GPUs"})
    hf_token: str = field(metadata={"help": "Hugging Face access token."})
    wandb_key: str = field(metadata={"help": "WandB API key."}) 
    wandb_host: str = field(
        default="http://wandb.artfacestudio.com",
        metadata={"help": "WandB host URL."}
    )
    project_name: str = field(
        default="count_qwen_vlm",
        metadata={"help": "WandB project name."}
    )

    pl_strategy: str = field(
        default="deepspeed_stage_2",
        metadata={"help": "pl_strategy"}
    )

    pl_precision: str = field(
        default="bf16-mixed",
        metadata={"help": "pl_precision"}
    )

    output_dir: str = "logs"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8 # number of steps before performing a backward/update pass
    gradient_checkpointing: bool = True
    evaluation_strategy: str = "epoch"
    optim: str = "adamw_torch_fused" # use fused adamw optimizer
    logging_steps: int = 5 # log every 5 steps
    save_strategy: str = "epoch"
    learning_rate: float = 2e-4 # learning rate, based on QLoRA paper
    bf16: bool = True
    max_grad_norm: float = 0.3 # max gradient norm based on QLoRA paper  1.0로 실험 for qwen
    # warmup_ratio: float = 0.03 # warmup ratio based on QLoRA paper
    warmup_steps: int = 50 # 
    lr_scheduler_type: str = "constant"
    push_to_hub: bool = False
    report_to: str = "wandb"
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    dataset_text_field: str = "" # need a dummy field for collator
    dataset_kwargs: dict = field(
        default_factory=lambda: {"skip_prepare_dataset": True} # important for collator
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    attn_implementation: str = field(default="sdpa")
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct") # Qwen/Qwen2.5-VL-32B-Instruct
    model_max_length: int = field(
        default=65536,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 64)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)