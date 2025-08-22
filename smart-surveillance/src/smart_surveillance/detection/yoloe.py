from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np


class YOLOEOpenVocabDetector:
    """
    Unified open-vocabulary detector/segmenter using Ultralytics YOLOE.

    Output format of detect():
    [
      {"cls": str, "score": float, "bbox": [x1,y1,x2,y2], "mask": Optional[np.ndarray]}
    ]
    """

    def __init__(self,
                 yoloe_model_path: str = "yoloe-11l-seg.pt",
                 score_threshold: float = 0.25,
                 max_dets: int = 50,
                 device: str = "cuda"):
        self.score_threshold = score_threshold
        self.max_dets = max_dets
        self.device = device

        # Lazy import to avoid hard dependency errors during docs build
        try:
            from ultralytics import YOLOE  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Ultralytics YOLOE not available. Please ensure ultralytics is installed "
                "and provides YOLOE class."
            ) from e

        self._YOLOE = YOLOE  # keep class ref for potential re-init
        self._model = YOLOE(yoloe_model_path)
        self._classes_set: Optional[List[str]] = None

    def _ensure_classes(self, queries: List[str]):
        # Set dynamic classes only when changed
        if self._classes_set == queries:
            return
        names = [str(x) for x in queries]
        try:
            # Prefer text positional encoding API if available (per user example)
            text_pe = self._model.get_text_pe(names)
            self._model.set_classes(names, text_pe)
        except Exception:
            # Fallback to simple set_classes if PE API absent
            try:
                self._model.set_classes(names)
            except Exception:
                pass
        self._classes_set = names

    def detect(self, pil_image, queries: List[str]) -> List[Dict[str, Any]]:
        self._ensure_classes(queries)

        # Run prediction
        results = self._model.predict(pil_image, conf=self.score_threshold, device=self.device, verbose=False)
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

        # Try to collect masks if segmentation head present
        masks_np: Optional[np.ndarray] = None
        try:
            if getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None:
                # r0.masks.data: (N, H, W) boolean/float mask tensor
                masks_np = r0.masks.data.detach().cpu().numpy()
        except Exception:
            masks_np = None

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            if float(s) < self.score_threshold:
                continue
            if len(dets) >= self.max_dets:
                break
            cls_name = queries[l] if 0 <= l < len(queries) else (
                getattr(r0, "names", {}).get(l, "object") if hasattr(r0, "names") else "object"
            )
            item: Dict[str, Any] = {
                "cls": str(cls_name),
                "score": float(s),
                "bbox": np.asarray(b, dtype=float).tolist(),
            }
            if masks_np is not None and i < masks_np.shape[0]:
                # Provide mask as binary numpy array (uint8 0/1)
                try:
                    m = (masks_np[i] > 0.5).astype(np.uint8)
                    item["mask"] = m
                except Exception:
                    pass
            dets.append(item)
        return dets


