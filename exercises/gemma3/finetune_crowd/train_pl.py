from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, HfArgumentParser
from lightning.fabric import Fabric

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class CustomArguments:
    nnodes: int = field(metadata={"help": "Number of Nodes"})
    ngpus: int = field(metadata={"help": "Number of GPUs"})
    pl_strategy: str = field(
        default="deepspeed_stage_2",
        metadata={"help": "pl_strategy"}
    )
    pl_precision: str = field(
        default="bf16-mixed",
        metadata={"help": "pl_precision"}
    )
    model_id: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Model ID for Hugging Face."}
    )
    processor_id: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Processor ID for Hugging Face."}
    )
    dataset_id: str = field(
        default="ty-kim/crowd_count",
        metadata={"help": "Dataset ID for training."}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size."}
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention implementation method."}
    )


def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for element in msg.get("content", []):
            if element.get("type") != "image":
                continue
            img_obj = element.get("image")

            if img_obj is None:
                continue

            if isinstance(img_obj, str):
                img_obj = PILImage.open(img_obj)

            image_inputs.append(img_obj.convert("RGB"))

    return image_inputs

def collate_fn(batch, processor):
    texts = []
    images = []

    for example in batch:
        image_inputs = process_vision_info(example["messages"])
        if not image_inputs:
            continue
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()

    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


class GemmaTrainer:
    def __init__(
            self,
            fabric,
            config):
        
        self.fabric = fabric
        self.cfg = config

        try:
            self.logger = self.fabric.logger
        except Exception as e:
            print(e)

        self.device = self.fabric.device
        self.init_weight_dtype()
        self.init_model_and_optimizer()
        self.prepare_dataset()


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
            self.processor = AutoProcessor.from_pretrained(self.cfg.processor_id, trust_remote_code=True, use_fast=False)
            # self.image_processor = self.processor.image_processor
        else:
            self.processor = processor
        
        if model is None:
            model_kwargs = dict(
                torch_dtype=torch.bfloat16,
                attn_implementation=self.cfg.attn_implementation,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(self.cfg.model_id, **model_kwargs)
        else:
            self.model = model

        self.set_model_requires_grad(self.cfg)
        self.init_optimizer()

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def set_model_requires_grad(self, config):
        """Set which parameters require gradients for training."""
        # Default: enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def init_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=5e-5,
            weight_decay=0.01
        )

    def prepare_dataset(self):
        dataset_dict = load_dataset(self.cfg.dataset_id, download_mode='force_redownload')
        self.train_dataset = dataset_dict['train']
        self.val_dataset = dataset_dict['test']

        self.train_dataset = self.train_dataset.cast_column("image_path", Image(decode=True))
        self.train_dataset = self.train_dataset.rename_column("image_path", "image")

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, self.processor))
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, self.processor))
        
        self.train_dataloader = self.fabric.setup_dataloaders(self.train_dataloader)
        self.val_dataloader = self.fabric.setup_dataloaders(self.val_dataloader)

    def train_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )

        return outputs.loss
        
    def evaluate(self, batch):
        pass
    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        for batch in self.train_dataloader:
            loss = self.train_step(batch)
            self.fabric.backward(loss)
            self.optimizer.step()


if __name__ == "__main__":
    parser = HfArgumentParser(CustomArguments)
    (config,) = parser.parse_args_into_dataclasses()

    fabric = Fabric(
        accelerator="cuda",
        num_nodes=config.nnodes,
        devices=config.ngpus,
        strategy=config.pl_strategy,
        precision=config.pl_precision,
    )

    fabric.launch()
    trainer = GemmaTrainer(fabric, config)
    trainer.train()