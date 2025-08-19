import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

from config import Configuration
from utils import train_collate_function, get_processor_with_new_tokens, get_model_with_resize_token_embeddings
import argparse
import albumentations as A


def get_augmentations(cfg):
    if "SmolVLM" in cfg.model_id:
        resize_size = 512
    else:
        resize_size = 896

    augmentations = A.Compose([
        A.Resize(height=resize_size, width=resize_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))
    return augmentations


def get_dataloader(processor, cfg):
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, device=cfg.device, transform=get_augmentations(cfg)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader

def train_model(model, optimizer, cfg, train_dataloader):
    print("Start training")
    global_step = 0
    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**{k: v.to(cfg.device) for k, v in batch.items()})
            loss = outputs.loss
            if idx % 100 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    return model


def run_training_phase(model, processor, cfg, train_dataloader, train_keys, phase_name="phase"):

    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in train_keys)

    model.train()
    model.to(cfg.device)

    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    wandb.init(
        project=cfg.project_name,
        name=f"{cfg.run_name}_{phase_name}" if hasattr(cfg, "run_name") else phase_name,
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader)

    wandb.finish()

from torch.utils.data import default_collate

class PaLiCollator:
    def __init__(self, processor, device, transform):
        self.processor = processor
        self.device = device
        self.transform = transform

    def __call__(self, examples):
        return train_collate_function(
            examples, processor=self.processor,
            device=self.device, transform=self.transform
        )

if __name__ == "__main__":
    cfg = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='Model ID on Hugging Face Hub')
    parser.add_argument('--dataset_id', type=str, help='Dataset ID on Hugging Face Hub')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--checkpoint_id', type=str, help='Model repo to push to the Hub')
    parser.add_argument('--include_loc_tokens', action='store_true', help='Include location tokens in the model.')
    parser.add_argument('--wandb_host', type=str, default="http://wandb.artfacestudio.com")
    parser.add_argument('--wandb_key', type=str, required=True)

    args = parser.parse_args()

    if args.model_id: cfg.model_id = args.model_id
    if args.dataset_id: cfg.dataset_id = args.dataset_id
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.learning_rate: cfg.learning_rate = args.learning_rate
    if args.epochs: cfg.epochs = args.epochs
    if args.checkpoint_id: cfg.checkpoint_id = args.checkpoint_id

    wandb.login(key=args.wandb_key, host=args.wandb_host)


    processor = AutoProcessor.from_pretrained(cfg.model_id)
    if args.include_loc_tokens:
        processor = get_processor_with_new_tokens(processor)

    

    if "SmolVLM" in cfg.model_id:
        model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype, device_map=cfg.device, _attn_implementation="eager")

    cfg.device = model.device
    train_dataloader = get_dataloader(processor=processor, cfg=cfg)

    if args.include_loc_tokens:
        # 추가한 토큰에 맞춰 임베딩수 재설정
        model = get_model_with_resize_token_embeddings(model, processor)

        # Stage 1: Training embed_tokens
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["embed_tokens"], phase_name="embed_only")

        # Stage 2: Fine-tuning embed_tokens + attn
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["embed_tokens", "attn"], phase_name="embed_attn")
    else:
        # Single-stage: Fine-tuning attn only
        run_training_phase(model, processor, cfg, train_dataloader, train_keys=["attn"], phase_name="attn_only")

    model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

    print("Train finished")