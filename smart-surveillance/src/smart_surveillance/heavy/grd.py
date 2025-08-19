# smart_surveillance/heavy/grd.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

class GroundingDINOWrapper:
    def __init__(self, model_id: str, device: str = "cuda",
                 box_th: float = 0.25, text_th: float = 0.2):
        self.device = device
        self.box_th = box_th
        self.text_th = text_th
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def refine(self, image: Image.Image, queries: List[str],
               seed_boxes: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Returns refined detections matching queries. If seed_boxes are given,
        you can optionally filter by overlap with seeds (left as simple post step).
        """
        inputs = self.proc(images=image, text=[queries], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        post = self.proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_th, text_threshold=self.text_th,
            target_sizes=[image.size[::-1]]
        )[0]
        boxes = post["boxes"].detach().cpu().numpy()
        scores = post["scores"].detach().cpu().numpy()
        labels = [str(x) for x in post.get("labels", queries)]
        dets = []
        for i, b in enumerate(boxes):
            dets.append({"cls": labels[i] if i < len(labels) else "object",
                         "score": float(scores[i]),
                         "bbox": [float(x) for x in b.tolist()]})
        # (선택) seed_boxes와 IoU 필터링을 추가 가능
        return dets
