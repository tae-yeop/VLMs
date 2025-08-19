from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    dataset_id: str = "ty-kim/license-detection"

    project_name: str = "gemma-3-4b-pt-object-detection-aug" # "SmolVLM-256M-Instruct-object-detection-aug"
    model_id: str = "google/gemma-3-4b-pt" # "HuggingFaceTB/SmolVLM-256M-Instruct"
    checkpoint_id: str = "ty-kim/gemma-3-4b-pt-object-detection-loc-tokens" # "sergiopaniego/SmolVLM-256M-Instruct-object-detection"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 # Change to torch.bfloat16 for "google/gemma-3-4b-pt" auto 여도 자동으로 bfloat16인거 같은데 

    batch_size: int = 2 # 8 for "google/gemma-3-4b-pt" 8을 쓰니깐 OOM
    learning_rate: float = 2e-05
    epochs = 2