import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


import debugpy

debugpy.listen(('0.0.0.0', 5678))

print("Waiting for debugger attach")
debugpy.wait_for_client()


import trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from data.data_qwen import make_supervised_data_module
from data.data_qwen_packed import make_supervised_data_module_packed
from argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


import os, torch
print("cuda available:", torch.cuda.is_available())
print("num devices:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.version.cuda:", torch.version.cuda)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    global local_rank

    # argparser + dataclass
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    # 쉘로 들어온 sys.argv를 파싱해서 각각 dataclass로 넣음
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Model Arguments:", model_args) # Model Arguments: ModelArguments(model_name_or_path='Qwen/Qwen-7B-Chat', tune_mm_llm=False, tune_mm_mlp=False, tune_mm_vision=False)
    print("Data Arguments:", data_args) # DataArguments(dataset_use='', video_max_frames=8, video_min_frames=4, data_flatten=False, data_packing=False, base_interval=2, max_pixels=451584, min_pixels=12544, video_max_frame_pixels=25088, video_min_frame_pixels=3136)
    print("Training Arguments:", training_args)

    # torchrun으로 돌리면 local_rank이 자동으로 설정됨
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)


    if "qwen2.5" in model_args.model_name_or_path.lower():

        print("Using Qwen2.5 VL model")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    
    print('model', type(model)) # 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration'
    print('image_processor', type(data_args.image_processor)) # 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    # grad 체크포인트시 별도 처리
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    print('tokenizer', type(tokenizer)) # 'transformers.models.qwen2_vl.tokenization_qwen2_vl.Qwen2VLTokenizer'

    
    # if torch.distributed.get_rank() == 0:
    #     model.visual.print_trainable_parameters()
    #     model.language_model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")