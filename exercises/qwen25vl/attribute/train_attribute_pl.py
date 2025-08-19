import torch
from torch.utils.data import DataLoader, Dataset

from lightning.fabric import Fabric

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer, AutoProcessor,
    HfArgumentParser
)
from dataclasses import dataclass, field

import time
import wandb
from wandb.integration.lightning.fabric import WandbLogger
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image


system_message = """You are a Vision Language Model specialized in interpreting visual data from person images.
Your task is to analyze the provided image and respond to queries with concise answers, usually a single word, number, or short phrase.
Focus on delivering accurate, detailed, and succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["text"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]


def load_image(image_path: str):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def process_vision_info(messages):
    images = []
    # videos = []  # not used for now
    for message in messages:
        if message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    images.append(load_image(content.get("image")))
                # elif content.get("type") == "video":
                #     videos.append(load_video(content.get("video")))
    return images


class Qwen25VLAttributeTrainer():
    def __init__(self, cfg, fabric, train_dataset):
        self.fabric = fabric
        self.cfg = cfg
        self.logger = self.fabric.logger
        self.device = self.fabric.device

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.image_processor = self.processor.image_processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            model_max_length=self.cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )

        self.model.config.use_cache = False
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            

        params_to_optimize = []
        total_params = 0
        param_txt = ""
        for pname, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                total_params += param.numel()
                param_txt += pname + '\n'

        optimizer_class = torch.optim.AdamW
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=getattr(self.cfg, "learning_rate", 2e-4),
            weight_decay=0.01,
        )

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)


        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=self.collate_fn)

        self.trainloader = self.fabric.setup_dataloaders(train_loader)


    def collate_fn(self, examples):
        """
        examples: list[dict]
        """

        # 문자와 이미지를 챗 형식으로 변환
        texts = [self.processor.apply_chat_template(example, tokenize=False) for example in examples]
        images = [process_vision_info(example) for example in examples]

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    def train_step(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                tensor_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                # Keep lists as-is (e.g., image metadata) because model forward handles them
                tensor_batch[key] = value
            else:
                tensor_batch[key] = value

        outputs = self.model(**tensor_batch)
        return outputs.loss

    
    def train(self):
        self.model.train()
        self.num_train_step = 0

        grad_accum_steps = max(1, getattr(self.cfg, "gradient_accumulation_steps", 1))
        max_grad_norm = getattr(self.cfg, "max_grad_norm", None)

        for epoch in range(getattr(self.cfg, "num_train_epochs", 1)):
            self.trainloader = tqdm(self.trainloader) if self.fabric.global_rank == 0 else self.trainloader

            for batch_idx, batch in enumerate(self.trainloader, start=1):
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


@dataclass
class CustomArguments:
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
        default="auto",
        metadata={"help": "pl_strategy"}
    )

    pl_precision: str = field(
        default="bf16-mixed",
        metadata={"help": "pl_precision"}
    )

    attn_implementation: str = field(default="sdpa")
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct") # Qwen/Qwen2.5-VL-32B-Instruct
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(CustomArguments)
    (config,) = parser.parse_args_into_dataclasses()

    wandb.login(key=config.wandb_key, host=config.wandb_host)
    wandb_logger = WandbLogger(
        project=config.project_name, 
        name=f"qwen_{time.strftime('%m%d')}", 
        config={**vars(config)},
    )


    deepspeed_config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": False,
                "offload_param": False,
            },
            "fp16": {"enabled": True},
        }

    
    fabric = Fabric(
        accelerator="cuda",
        num_nodes=config.nnodes,
        devices=config.ngpus,
        strategy=config.pl_strategy,
        precision=config.pl_precision,
        loggers=wandb_logger
    )

    fabric.launch()

    dataset = load_dataset("csv", data_files={"train": "/purestorage/AILAB/AI_4/datasets/PETA/PETA_train_dataset.csv", "validation":"/purestorage/AILAB/AI_4/datasets/PETA/PETA_validation_dataset.csv"})

    # Convert each CSV row into a chat message list
    train_chats = [format_data(sample) for sample in dataset["train"]]
    eval_chats = [format_data(sample) for sample in dataset["validation"]]

    print(train_chats[0])

    # Wrap chats with a simple PyTorch Dataset instead of converting to HF Dataset
    class ChatDataset(Dataset):
        def __init__(self, chats):
            self.chats = chats

        def __len__(self):
            return len(self.chats)

        def __getitem__(self, idx):
            return self.chats[idx]

    train_dataset = ChatDataset(train_chats)
    eval_dataset = ChatDataset(eval_chats)

    trainer = Qwen25VLAttributeTrainer(
        cfg=config,
        fabric=fabric,
        train_dataset=train_dataset,
    )

    trainer.train()




