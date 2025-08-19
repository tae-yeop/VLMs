# https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/
import sys
import time
from turtle import position
from typing import Dict, List, Tuple, Optional
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from lightning.fabric import Fabric
import lightning as L

from wandb.integration.lightning.fabric import WandbLogger
import wandb
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, Qwen2VLImageProcessor, Trainer, BitsAndBytesConfig, HfArgumentParser
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from datasets import load_dataset

# import debugpy

# debugpy.listen(('0.0.0.0', 5678))

# print("Waiting for debugger attach")
# debugpy.wait_for_client()


def preprocess_qwen_2_visual(
    sources,
    tokenizer,
    grid_thw_image=[],
    grid_thw_video=[],
):
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    messages = json.loads(sources[0])

    system_message = next((c.get('text') for c in messages[0].get('content', []) if c.get('type') == 'text'), None)

    system_message_dict = [{"role": "system", "content": system_message}]

    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        system_message_dict
    )
    target += [IGNORE_INDEX] * len(input_id)


    pair = to_qwen_style_pair(messages[1], messages[2])

    for conv in pair['messages']:
        role = conv['role']
        content = conv['content']

        if role == "user":
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>"
                        * grid_thw_image[visual_replicate_index_image]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    visual_replicate_index_image += 1

                new_parts.append(parts[-1])
                content = "".join(new_parts)

        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id

        if role in ["user", "system"]:
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            target_mask = encode_id.copy()
            target_mask[:3] = [IGNORE_INDEX] * 3
            target += target_mask

    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    input_ids.append(input_id)
    targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets
    )

class QwenTrainer:
    def __init__(
            self,
            fabric,
            config,
            hf_trainset=None,
            hf_valset=None):
        
        self.fabric = fabric
        self.cfg = config

        try:
            self.logger = self.fabric.logger
        except Exception as e:
            print(e)

        self.device = self.fabric.device
        self.init_weight_dtype()
        self.init_model_and_optimizer()
        self.prepare_dataset(hf_trainset, hf_valset)

    def init_weight_dtype(self):
        precision_str = self.cfg.pl_precision

        if '16' in precision_str or 'transformer-engine' in precision_str:
            if 'bf' in precision_str:
                self.weight_dtype = torch.bfloat16
            else:
                self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

    def init_model_and_optimizer(self, processor=None, model=None):
        if processor is None:
            self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
            self.image_processor = self.processor.image_processor
        else:
            self.processor = processor
        
        if model is None:   
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.cfg.attn_implementation,
            )
        else:
            self.model = model

        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            model_max_length=self.cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_model_requires_grad(self.cfg)
        self.init_optimizer()
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def set_model_requires_grad(self, model_args):
        if model_args.tune_mm_vision:
            for n, p in self.model.visual.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.visual.named_parameters():
                p.requires_grad = False

        if model_args.tune_mm_mlp:
            for n, p in self.model.visual.merger.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.visual.merger.named_parameters():
                p.requires_grad = False

        if model_args.tune_mm_llm:
            for n, p in self.model.model.named_parameters():
                p.requires_grad = True
            self.model.lm_head.requires_grad = True
        else:
            for n, p in self.model.model.named_parameters():
                p.requires_grad = False
            self.model.lm_head.requires_grad = False



    def init_optimizer(self):
        params_to_optimize = []

        params = self.get_trainable_params()
        if len(params) == 0:
            # Fallback: if nothing marked as trainable, train all params
            params_to_optimize = list(self.model.parameters())
        else:
            params_to_optimize.extend(params)

        optimizer_class = None
        if 'offload' in self.cfg.pl_strategy:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer_class = DeepSpeedCPUAdam
        else:
            optimizer_class = torch.optim.AdamW

        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=getattr(self.cfg, "learning_rate", 2e-4),
            weight_decay=0.01,
        )

        return self.optimizer

    def get_trainable_params(self):
        params = []
        total_params = 0
        param_txt = ""
        for pname, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
                total_params += param.numel()
                param_txt += pname + '\n'

        return params

    def prepare_dataset(self, hf_trainset=None, hf_valset=None):

        train_dataset = CrowdCountingDataset(
            hf_trainset,
            self.tokenizer,
            self.image_processor,
            self.cfg
        )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)


        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_collator,
        )

        if hf_valset is not None:
            val_dataset = CrowdCountingDataset(
                hf_valset,
                self.tokenizer,
                self.image_processor,
                self.cfg
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.per_device_val_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate,
            )


    def train_step(self, batch):
        # Accept either prebuilt tuple or raw collated dict
        # if isinstance(batch, dict):
        #     input_ids, attention_mask, pixel_values, image_grid_thw, labels = self.process_batch_with_processor(batch)
        # else:
        #     input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        
        # input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        image_grid_thw = batch["image_grid_thw"].to(self.device)
        position_ids = batch["position_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            position_ids=position_ids,
        )

        loss = outputs.loss
        return loss

    def process_batch_with_processor(self, batch):
        images = batch["image"]  # list of PIL images
        counts = batch["counts"]  # tensor of shape (B,)

        # Build chat-style prompts with ground-truth answer for supervised LM loss
        messages_list = []
        for i, image in enumerate(images):
            answer_text = str(int(counts[i].item()))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": "Count the number of people in the image. Answer with a single integer.",
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
            ]
            messages_list.append(messages)

        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in messages_list
        ]

        processed = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = processed["input_ids"].to(self.device)
        attention_mask = processed["attention_mask"].to(self.device)
        pixel_values = processed["pixel_values"].to(self.device, dtype=self.weight_dtype)
        image_grid_thw = processed["image_grid_thw"].to(self.device)
        # Supervised LM loss over full sequence (simple, effective baseline)
        labels = input_ids.clone()

        return input_ids, attention_mask, pixel_values, image_grid_thw, labels

    def evaluate(self, batch):
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids
            in zip(input_ids, generated_ids)]
        
        generated_suffixes = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        scores = []
        for generated_suffix, suffix in zip(generated_suffixes, suffixes):
            score = edit_distance(generated_suffix, suffix)
            score = score / max(len(generated_suffix), len(suffix))
            scores.append(score)
            print("generated_suffix", generated_suffix)
            print("suffix", suffix)
            print("score", score)

        score = sum(scores) / len(scores)
        self.log("val_edit_distance", score, prog_bar=True, logger=True, batch_size=self.config.get("batch_size"))
        return scores


    def train(self):
        self.model.train()

        self.num_train_step = 0
        grad_accum_steps = max(1, getattr(self.cfg, "gradient_accumulation_steps", 1))
        max_grad_norm = getattr(self.cfg, "max_grad_norm", None)
        for epoch in range(getattr(self.cfg, "num_train_epochs", 1)):
            self.train_loader = tqdm(self.train_loader) if self.fabric.global_rank == 0 else self.trainloader
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                loss = self.train_step(batch)

                self.num_train_step += 1
                if hasattr(self, "logger") and self.fabric.global_rank == 0:
                    try:
                        self.logger.log_metrics({'loss': float(loss.detach().item())}, step=self.num_train_step)
                    except Exception:
                        pass

                # Scale and accumulate gradients
                scaled_loss = loss / grad_accum_steps
                self.fabric.backward(scaled_loss)
                if batch_idx % grad_accum_steps == 0:
                    if max_grad_norm is not None and max_grad_norm > 0:
                        self.fabric.clip_gradients(self.model, self.optimizer, max_norm=max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Periodic lightweight eval on the current batch (optional)
                if (self.num_train_step % getattr(self.cfg, "logging_steps", 50) == 0) and hasattr(self, "valloader") and self.valloader is not None:
                    try:
                        val_batch = next(iter(self.valloader))
                        self.evaluate(val_batch)
                    except Exception:
                        pass


import os
import copy
import numpy as np
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
from torch.utils.data import Dataset
import torch, torchvision.transforms as T
from datasets import load_dataset, Image
# Support running both as a package module and as a standalone script
try:
    from .rope2d import get_rope_index_25  # when executed as part of the package
except Exception:
    try:
        from qwen25vl.crowd_counting.rope2d import get_rope_index_25  # absolute import fallback
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from rope2d import get_rope_index_25  # local directory fallback




if __name__ == '__main__':
    parser = HfArgumentParser(CustomArguments)
    (config,) = parser.parse_args_into_dataclasses()
    # training_args.remove_unused_columns = False
    


    print('========config=========')
    print(config)

    wandb.login(key=config.wandb_key, host=config.wandb_host)
    wandb_logger = WandbLogger(
        project=config.project_name, 
        name=f"qwen_{time.strftime('%m%d')}", 
        config={**vars(config)},
    )

    train_dataset = load_dataset(
        "ty-kim/total_crowd_count2", 
        split="train",
        download_mode="force_redownload")
        # use_auth_token=config.hf_token,

    train_dataset = train_dataset.cast_column("image_path", Image(decode=True))
    train_dataset = train_dataset.rename_column("image_path", "image")   # 이제 'image' 컬럼 존재

    # Keep images as PIL for the Qwen processor; only convert numeric fields
    def _tf(ex):
        # ex["image"] stays as PIL.Image
        ex["points"] = torch.tensor(ex["points"], dtype=torch.float32)
        return ex

    train_dataset.set_transform(_tf)
    # Keep 'image' column in outputs while formatting only tensor fields
    train_dataset.set_format(type="torch", columns=["points", "counts"], output_all_columns=True)


    deepspeed_config = {
            "train_micro_batch_size_per_gpu": 1,  # 적절한 값으로 설정
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": False,
                "offload_param": False,
            },
            "fp16": {"enabled": True},
        }

    # Map strategy string to Fabric strategy instance when needed
    strategy = config.pl_strategy
    if isinstance(strategy, str) and strategy.startswith("deepspeed_stage_"):
        stage = 2 if "_2" in strategy else 3
        offload = "offload" in strategy
        strategy = DeepSpeedStrategy(stage=stage, offload_optimizer=offload, offload_parameters=offload)

    fabric = Fabric(
        accelerator="cuda",
        num_nodes=config.nnodes,
        devices=config.ngpus,
        strategy=config.pl_strategy,
        precision=config.pl_precision,
        loggers=wandb_logger
    )

    fabric.launch()

    trainer = QwenTrainer(
        fabric=fabric,
        config=config,
        hf_trainset=train_dataset,
        hf_valset=None # Assuming no validation loader for now
    )

    trainer.train()