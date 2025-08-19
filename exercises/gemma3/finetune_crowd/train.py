import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig, HfArgumentParser
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
import argparse
import time
import wandb


# import debugpy

# debugpy.listen(('0.0.0.0', 5678))

# print("Waiting for debugger attach")
# debugpy.wait_for_client()

@dataclass
class CustomArguments:
    # 기본값 없는걸 앞에 몰아서 써야함. 안그럼 에러
    # 이게 싫으면 모든 항목에 기본값을 deafualt=None을 주도록 함
    hf_token: str = field(metadata={"help": "Hugging Face access token."})
    wandb_key: str = field(metadata={"help": "WandB API key."}) 
    model_id: str = field(
        default="google/gemma-3-1b-pt",
        metadata={"help": "Model ID for Hugging Face."}
    )
    wandb_host: str = field(
        default="http://wandb.artfacestudio.com",
        metadata={"help": "WandB host URL."}
    )
    project_name: str = field(
        default="crowd_counting",
        metadata={"help": "WandB project name."}
    )


@dataclass
class MySFTConfig(SFTConfig):
    # 반드시 타입을 같이 넣어야 HfArgumentParser에서 동작함
    # 안그럼 그냥 TraingArgumnet 디폴트가 만들어져서 잡힘
    output_dir: str = "output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4 # number of steps before performing a backward/update pass
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused" # use fused adamw optimizer
    logging_steps: int = 5 # log every 5 steps
    save_strategy: str = "epoch"
    learning_rate: float = 2e-4 # learning rate, based on QLoRA paper
    bf16: bool = True
    max_grad_norm: float = 0.3 # max gradient norm based on QLoRA paper
    warmup_ratio: float = 0.03 # warmup ratio based on QLoRA paper
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

from PIL import Image as PILImage
import io, os

def process_vision_info(messages):
    # messages : list of dict, 각 turn이 dict
    image_inputs = []
    for msg in messages: # 각 turn 을 살펴봄
        for element in msg.get("content", []):
            # print(element)
            if element.get("type") != "image":
                continue
            img_obj = element.get("image")

            if img_obj is None:
                continue
            
            # print(type(img_obj))
            if isinstance(img_obj, str):
                # print('dsad')
                img_obj = PILImage.open(img_obj)

            # print(img_obj)
            image_inputs.append(img_obj.convert("RGB"))

    return image_inputs

if __name__ == "__main__":
    parser = HfArgumentParser((CustomArguments, MySFTConfig))
    custom_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False # important for collator

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        wandb.login(key=custom_args.wandb_key, host=custom_args.wandb_host)
        wandb.init(
            project=custom_args.project_name,
            name=f"{custom_args.project_name}_{time.strftime('%m%d')}",
            config={**vars(custom_args), **training_args.to_dict()},
        )

    # Load dataset from the hub
    dataset_dict = load_dataset("ty-kim/crowd_count")
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['test']

    model_id = "google/gemma-3-4b-pt"
    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
        device_map=None, # Let torch decide how to load the model
    )

    # BitsAndBytesConfig int-4 config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing=training_args.gradient_checkpointing)

    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens"
        ]
    )

    # LLM은 Collator 클래스 제공하는게 있어서 그걸 쓰면됨.
    def collate_fn(examples):
        """
        exampels : list of dict로 이뤄진 배치, [{'messages': []}, {'messages': []}, {'messages': []},....]
        """
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            if not image_inputs:
                continue
            text = processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens for not being used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch
    

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn
    )


    # Start training, the model will be automatically saved to the Hub and the output directory
    trainer.train()

    # Save the final model again to the Hugging Face Hub
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        trainer.save_model()

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    from peft import PeftModel

    # Load Model base model
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

        # Merge LoRA and base model and save
        peft_model = PeftModel.from_pretrained(model, training_args.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

        processor = AutoProcessor.from_pretrained(training_args.output_dir)
        processor.save_pretrained("merged_model")


