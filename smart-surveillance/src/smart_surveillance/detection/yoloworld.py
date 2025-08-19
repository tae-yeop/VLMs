# smart_surveillance/detection/yoloworld.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

class YOLOWorldDetector:
    """
    Open-vocab detector interface.
    If YOLO-World backend is unavailable, falls back to OWL-ViT via HF.
    Output format:
    [
      {"cls": "person", "score": 0.88, "bbox": [x1,y1,x2,y2]},
      ...
    ]
    """
    def __init__(self, backend: str = "owlvit",
                 yoloworld_model_path: str | None = None,
                 owlvit_model_id: str = "google/owlv2-base-patch16",
                 score_threshold: float = 0.25,
                 max_dets: int = 50,
                 device: str = "cuda"):
        self.backend = backend
        self.score_threshold = score_threshold
        self.max_dets = max_dets
        self.device = device

        self._impl = None
        if backend == "yoloworld":
            # TODO: 연결할 YOLO-World 백엔드(TensorRT/ONNX/PyTorch)를 여기에 붙이세요.
            # 예: onnxruntime / TensorRT 엔진 로더
            # self._impl = YOLOWORLD_ONNX(yoloworld_model_path, device)
            # 1) 시도: Ultralytics YOLO-World (있으면 사용)
            try:
                from ultralytics import YOLOWorld as UltralyticsYOLOWorld  # type: ignore
                # Common model names: yolov8s-worldv2.pt, yolov8m-worldv2.pt, yolov8l-worldv2.pt
                model_path = yoloworld_model_path or "yolov8s-worldv2.pt"
                self._impl = UltralyticsYOLOWorld(model_path)
                self.backend = "yoloworld_ultralytics"
            except Exception as e:
                # 2) 실패 시 폴백: OWL-ViT
                print("[WARN] YOLO-World backend not configured; falling back to OWL-ViT.")
                print("[HINT] To enable YOLO-World, install 'ultralytics' and provide a model path via cfg or --yoloworld-model.")
                backend = "owlvit"
                self.backend = "owlvit"

        if backend == "owlvit":
            from transformers import Owlv2ForObjectDetection, AutoProcessor
            import torch
            self._proc = AutoProcessor.from_pretrained(owlvit_model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(owlvit_model_id).to(device)
            self._torch = torch

    def detect(self, pil_image, queries: List[str]) -> List[Dict[str, Any]]:
        # If we have fallen back or explicitly using OWL-ViT
        if self.backend == "owlvit":
            return self._detect_owlvit(pil_image, queries)
        # Ultralytics YOLO-World dynamic open-vocab
        if self.backend == "yoloworld_ultralytics" and self._impl is not None:
            # Ultralytics accepts PIL/numpy directly
            # Set dynamic categories to queries
            try:
                self._impl.set_classes(queries)
            except Exception:
                pass
            results = self._impl.predict(pil_image, conf=self.score_threshold, device=self.device, verbose=False)
            if not results:
                return []
            r0 = results[0]
            dets: List[Dict[str, Any]] = []
            try:
                boxes = r0.boxes.xyxy.cpu().numpy()
                scores = r0.boxes.conf.cpu().numpy()
                labels = r0.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                return []
            for b, s, l in zip(boxes, scores, labels):
                if s < self.score_threshold:
                    continue
                if len(dets) >= self.max_dets:
                    break
                cls_name = queries[l] if 0 <= l < len(queries) else (getattr(r0, "names", {}).get(l, "object") if hasattr(r0, "names") else "object")
                dets.append({
                    "cls": str(cls_name),
                    "score": float(s),
                    "bbox": np.asarray(b, dtype=float).tolist(),
                })
            return dets
        # elif self.backend == "yoloworld": return self._impl.detect(...)
        raise NotImplementedError("No backend configured")

    def _detect_owlvit(self, pil_image, queries: List[str]) -> List[Dict[str, Any]]:
        # OWL-ViT expects list of texts (categories)
        inputs = self._proc(text=[queries], images=pil_image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        # Post-process to XYXY in original image size
        target_sizes = self._torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self._proc.post_process_object_detection(outputs=outputs, threshold=self.score_threshold,
                                                           target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"].detach().cpu().numpy()

        dets: List[Dict[str, Any]] = []
        for b, s, l in zip(boxes, scores, labels):
            if s < self.score_threshold: continue
            if len(dets) >= self.max_dets: break
            dets.append({"cls": queries[int(l)] if int(l) < len(queries) else "object",
                         "score": float(s),
                         "bbox": b.astype(float).tolist()})
        return dets
