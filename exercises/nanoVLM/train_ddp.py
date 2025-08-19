import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import os
import re
import wandb
import argparse
import numpy as np
from tqdm import tqdm

import math
import time
import torch
import wandb
import numpy
import random
import argparse
import torch.optim as optim

from dataclasses import asdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from data.collators import VQACollator, MMStarCollator
from data.datasets import MMStarDataset, VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config
import models.utils as utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{train_cfg.batch_size}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    date = time.strftime("%m%d")

    return f"nanoVLM_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)

    test_ds = load_dataset(train_cfg.test_dataset_path)
    train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatinated datasets

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(train_ds.select(range(train_size, total_samples)), tokenizer, image_processor)
    test_dataset = MMStarDataset(test_ds['val'], tokenizer, image_processor)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=train_cfg.mmstar_batch_size, 
        shuffle=False, 
        collate_fn=mmstar_collator,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )

    return train_loader, val_loader, test_loader


def test_mmstar(model, tokenizer, test_loader, device):
    model.eval()
    total_examples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            gen = model.generate(input_ids, image, attention_mask)
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            is_correct = utils.check_multiple_choice_with_regex(model_output, correct_answer)
            
            total_examples += len(is_correct)
            if is_correct:
                correct_predictions += sum(is_correct)
    model.train()
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    return accuracy

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def resolve_root_node_address(nodes: str) -> str:
    """The node selection format in SLURM supports several formats.

    This function selects the first host name from

    - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
    - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
    - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root

    """
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  # Take the first node of every node range
    nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)  # handle special case where node range is single number
    return nodes.split(" ")[0].split(",")[0]

def get_main_address():
    root_node = os.environ.get("MASTER_ADDR")
    if root_node is None:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node

    return root_node

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 실행시 torchrun or torch.distributed.launch --
        args.rank = int(os.environ["RANK"]) # dist.get_rank()
        args.world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
        args.local_rank = int(os.environ['LOCAL_RANK']) # args.rank % torch.cuda.device_count()
    elif 'SLURM_PROCID' in os.environ:
        # 스크립트 실행시
        # #SBATCH --gres=gpu:x
        # #SBATCH --ntasks-per-node=x
        # python train.py
        # python train.py를 x번 돌리는 경우
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.node_rank = int(os.environ['SLURM_NODEID'])

        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)

    elif torch.cuda.is_available():
        # 스크립트 실행시 
        # 슬럼 옵션 사용하지 않을시
        # 이때는 torchrun or torch.distributed.launch도 안쓴다고 가정
        print('Will run the code on one GPU.')
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        is_master = True
    else:
        # GPU 마저도 안쓰는 경우 그냥 CPU
        print('Not using distributed mode')
        sys.exit(1)

    os.environ['MASTER_ADDR'] = get_main_address()
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    is_master = args.rank == 0
    dist.barrier()

    setup_for_distributed(args.rank == 0)
    print('| distributed init (rank {}): {}'.format(args.rank, os.environ['MASTER_ADDR']), flush=True)

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    args = parser.parse_args()
    # print(args)

    init_distributed_mode(args)

    print(args)

    fix_random_seeds(args.seed+args.rank)

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False


    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    total_dataset_size = len(train_loader.dataset)

    if train_cfg.log_wandb:
        run_name = get_run_name(train_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        
        wandb.login(key=train_cfg.wandb_key, host=train_cfg.wandb_host)
        run = wandb.init(
            project=train_cfg.wandb_project_name,
            entity=train_cfg.wandb_entity,
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name
        )

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
    print(f"Training summary: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch, batch size {train_cfg.batch_size}")
    print(f"Validation summary: {len(val_loader.dataset)} samples, {len(val_loader)} batches/epoch, batch size {train_cfg.batch_size}")

    param_groups = [
        {'params': model.MP.parameters(), 'lr': train_cfg.lr_mp},
        {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}
    ]
    optimizer = optim.AdamW(param_groups)

    model.to(args.device)

    epoch_times = []
    best_accuracy = 0
    global_step = 0
    for epoch in range(train_cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0

        for batch in train_loader:
            batch_start_time = time.time()
            images = batch["image"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            loss.backward()

            adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, len(train_loader) * train_cfg.epochs)
            adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, len(train_loader) * train_cfg.epochs)
            optimizer.param_groups[0]['lr'] = adj_lr_mp
            optimizer.param_groups[1]['lr'] = adj_lr_backbones

            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            num_tokens += images.shape[0] * ((images.shape[2] / vlm_cfg.vit_patch_size) ** 2) / (vlm_cfg.mp_pixel_shuffle_factor ** 2)

            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration

            if train_cfg.eval_in_epochs and global_step % 250 == 0:
                model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    epoch_accuracy = test_mmstar(model, tokenizer, test_loader, args.device)
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["image"].to(args.device)
                        input_ids = batch["input_ids"].to(args.device)
                        labels = batch["labels"].to(args.device)
                        attention_mask = batch["attention_mask"].to(args.device)

                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _, loss = model(input_ids, images, attention_mask, targets=labels)

                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                model.train()

                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    model.save_pretrained(save_directory=vlm_cfg.vlm_checkpoint_path)
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {epoch_accuracy:.4f} | Saving checkpoint to {vlm_cfg.vlm_checkpoint_path}")
                else:
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                if train_cfg.log_wandb:
                    run.log(
                        {"accuracy": epoch_accuracy,
                        "val_loss": avg_val_loss}, step=global_step
                    )

            if train_cfg.log_wandb:
                run.log({"batch_loss": batch_loss,
                "toknes_per_second": tokens_per_second}, step=global_step)

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if train_cfg.log_wandb:
            run.log({"epoch_loss": avg_train_loss,
                     "epoch_duration": epoch_duration,
                     "epoch_tokens_per_second": epoch_tokens_per_second})

        print(f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

        # Summary Statistics
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        accuracy = test_mmstar(model, tokenizer, test_loader, args.device)
        print(f"MMStar Accuracy: {accuracy:.4f}")

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["mmstar_acc"] = accuracy
            run.finish()



