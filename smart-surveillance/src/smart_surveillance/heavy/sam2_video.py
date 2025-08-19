# smart_surveillance/heavy/sam2_video.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import os


class SAM2Video:
    """
    Wrapper for SAM2 video segmentation/tracking.
    Tries multiple import paths; if unavailable, returns empty masks.
    """
    def __init__(self, model_id: str, device: str = "cuda", dtype: torch.dtype | None = None):
        self.device = device
        self.available = False
        try:
            try:
                from sam2 import SAM2VideoPredictor  # type: ignore
                Predictor = SAM2VideoPredictor
            except Exception:
                # Some builds expose the class under a module
                from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
                Predictor = SAM2VideoPredictor
            # Choose a safe default dtype: float16 on CUDA, float32 otherwise
            if dtype is None:
                env_dtype = os.environ.get("SAM2_DTYPE", "").lower()
                if env_dtype in ("fp32", "float32"):  # override by env
                    dtype = torch.float32
                elif env_dtype in ("bf16", "bfloat16"):
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16 if device.startswith("cuda") else torch.float32

            self.model = Predictor.from_pretrained(model_id)
            self.model.to(device=torch.device(device), dtype=dtype)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"[WARN] SAM2 not available: {e}. Masks will be empty.")

    @torch.no_grad()
    def track(self, video_path: str, init_frame_idx: int, init_box_xyxy: List[float]) -> Dict[int, np.ndarray]:
        if not self.available:
            return {}  # no masks
        try:
            state = self.model.init_state(video_path)
        except RuntimeError as e:
            # dtype mismatch (e.g., bfloat16) → retry by casting model to float16
            if "bfloat16" in str(e).lower():
                try:
                    self.model.to(dtype=torch.float16)
                    state = self.model.init_state(video_path)
                except Exception:
                    raise
            else:
                raise
        # advance to init frame
        try:
            for _ in range(max(0, init_frame_idx)):
                _ = self.model.propagate_in_video(state, step=1)
        except TypeError:
            gen = self.model.propagate_in_video(state)
            for _ in range(max(0, init_frame_idx)):
                try:
                    next(gen)
                except StopIteration:
                    break
        prompt = {"boxes": [[float(x) for x in init_box_xyxy]]}
        frame_idx, obj_ids, masks = self.model.add_new_points_or_box(state, prompts=prompt)
        tracked: Dict[int, np.ndarray] = {}
        if masks is not None and len(masks) > 0:
            tracked[int(frame_idx)] = masks[0].detach().cpu().numpy()
        for frame_idx, obj_ids, masks in self.model.propagate_in_video(state):
            if masks is not None and len(masks) > 0:
                tracked[int(frame_idx)] = masks[0].detach().cpu().numpy()
        return tracked
