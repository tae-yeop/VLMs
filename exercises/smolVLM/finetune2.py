import os
import time
import re
import argparse
import numpy as np
import random
import inspect
from tqdm import tqdm

import wandb
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.distributed as dist

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset

from config import TrainConfig

class SmolVLMTrainer():
    def __init__(self, fabric, cfg, train_loader, val_loader=None, test_loader=None):
        self.fabric = fabric
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        try:
            self.logger = self.fabric.logger
        except Exception as e:
            print(e)


        self.init_weight_dtype()
        self._build_model()
        self.setup_fabric()

    def init_weight_dtype(self):
        precision_str = self.cfg.pl_precision
        if '16' in precision_str or 'transformer-engine' in precision_str:
            if 'bf' in precision_str:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def _build_model(self):
            
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.dtype,
            _attn_implementation="flash_attention_2",
        )

        if not self.cfg.vision_finetune:
            for param in self.model.model.vision_model.parameters():
                param.requires_grad = False

        params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [
            {'params':params, 'lr': self.cfg.learning_rate}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.fabric.device=='cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(param_groups, betas=(self.cfg.beta1, self.cfg.beta2), **extra_args)


    def setup_fabric(self):
        if self.train_loader is not None:
            self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
        
        if self.val_loader is not None:
            self.val_loader = self.fabric.setup_dataloaders(self.val_loader)

        if self.test_loader is not None:
            self.test_loader = self.fabric.setup_dataloaders(self.test_loader)

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def generate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.train_loader:
                out = self.model(**batch)
                print(out)
                break
    def train(self):
        self.model.train()
        self.num_train_step = 0
        self.label_smoother = LabelSmoother(epsilon=0.1)

        for _ in range(self.cfg.num_epochs):
            self.train_loader = tqdm(self.train_loader, desc="Training") if self.fabric.is_global_zero else self.train_loader
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(**batch)
                self.fabric.log('loss', out.loss, step=self.num_train_step)
                if self.label_smoother is not None:
                    out.loss = self.label_smoother(out, batch["labels"])
                self.fabric.backward(out.loss, model=self.model)
                self.optimizer.step()
                self.num_train_step += 1


from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
class VQADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image']
        question = item["question"]
        answer = item["multiple_choice_answer"]


        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            print(f"Error processing image at index {idx}")
            image = torch.zeros(3, 384, 384)

        message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
        ]

        return {
            "image": image,
            "message": message
        }

class VQACollator():
    def __init__(self, processor):
        self.processor = processor

        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        messages = [
            self.processor.apply_chat_template(item["message"], add_generation_prompt=False).strip() 
            for item in batch
        ]

        batch = processor(text=messages, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch


def get_dataloader(cfg, processor):
    ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
    split_ds = ds["validation"].train_test_split(test_size=0.5)
    train_ds = split_ds["train"]

    train_dataset = VQADataset(train_ds)
    vqa_collator = VQACollator(processor)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    cfg = TrainConfig()

    wandb.login(key=cfg.wandb_key, host=cfg.wandb_host)
    wandb_logger = WandbLogger(
        project=cfg.wandb_project, 
        name=cfg.wandb_run_name, 
        config=asdict(cfg))

    fabric = Fabric(
            accelerator='cuda', 
            num_nodes=args.nnodes,
            devices=args.ngpus, 
            strategy=cfg.pl_strategy,
            precision=cfg.pl_precision,
            loggers=wandb_logger
    )

    fabric.launch()

    
    model_id = "HuggingFaceTB/SmolVLM-Base"
    processor = AutoProcessor.from_pretrained(model_id)

    train_loader = get_dataloader(cfg, processor)


    trainer = SmolVLMTrainer(fabric, cfg, train_loader)

    trainer.train()