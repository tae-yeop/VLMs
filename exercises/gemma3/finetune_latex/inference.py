from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import requests
from PIL import Image


if __name__ == '__main__':

    model_id = "google/gemma-3-4b-pt" 

    model_kwargs = dict(
        attn_implementation="flash_attention_2",  # more efficient attention mechanism for Ampere GPU
        torch_dtype=torch.bfloat16,  
        device_map="auto",  # Let torch decide how to load the model
    )

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")


    print(type(model))
    print(type(processor))