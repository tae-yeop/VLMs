from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers.image_utils import load_image

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                                    torch_dtype=torch.bfloat16,
                                                    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)