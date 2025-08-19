import os
import sys

from transformers import AutoProcessor, Idefics3ForConditionalGeneration

if __name__ == "__main__":

    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    checkpoint = None

    processor = AutoProcessor.from_pretrained(model_id)
    if checkpoint is not None:
        model = Idefics3ForConditionalGeneration(
            checkpoint,
            torch_dtype=torch.bloat16,
            device_map='cuda'
        )
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )

    