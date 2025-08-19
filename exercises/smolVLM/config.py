from dataclasses import dataclass

@dataclass
class TrainConfig:
    wandb_project = 'smolvlm-vqav2'
    wandb_run_name = 'run-pl' # 'run' + str(time.time())
    wandb_entity: str = "ailab" # Indicate the entity to log to in wandb
    wandb_host = # 
    wandb_key = # 
    pl_strategy = 'deepspeed_stage_2'
    pl_precision = 'bf16-mixed'
    # train
    num_epochs = 2
    batch_size = 16 # 32는 OOM
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    # model
    model_id = 'HuggingFaceTB/SmolVLM-Base'
    use_qlora = False
    vision_finetune = False