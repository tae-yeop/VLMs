# src/smart_surveillance/ingestion/video_loader.py
from __future__ import annotations
from typing import Iterator, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from smart_surveillance.configs import IngestionConfig

class VideoLoader:
    """
    Minimal MP4 frame loader.
    - Returns frames in BGR (OpenCV default)
    - Yields (meta, frame) where meta contains index, pts_ms, fps, size
    - Respects IngestionConfig: every_nth_frame, max_frames, resize_width/height
    Future: RTSP adapter can reuse the same interface.
    """

    def __init__(self, source_uri: str, cfg: IngestionConfig):
        """
        Args:
          source_uri: e.g. "file:///path/video.mp4" or plain path "/path/video.mp4"
          cfg: ingestion config
        """
        self.source_uri = _normalize_source(source_uri)
        self.cfg = cfg

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source_uri)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source_uri}")
        return cap

    def iterate_frames(self) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
        """
        Yields:
          meta: {"index", "pts_ms", "fps", "width", "height", "source"}
          frame: np.ndarray (H,W,3) in BGR
        """
        cap = self._open()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            every = max(1, int(self.cfg.every_nth_frame or 1))
            max_frames: Optional[int] = (
                int(self.cfg.max_frames) if (self.cfg.max_frames or 0) > 0 else None
            )

            idx = 0
            yielded = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if idx % every != 0:
                    idx += 1
                    continue

                # Resize if requested
                if self.cfg.resize_width and self.cfg.resize_height:
                    frame = cv2.resize(
                        frame,
                        (int(self.cfg.resize_width), int(self.cfg.resize_height)),
                        interpolation=cv2.INTER_AREA,
                    )

                # PTS(ms) fallback if driver doesn't provide it
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not (isinstance(pts_ms, (int, float)) and pts_ms == pts_ms):
                    pts_ms = (idx / fps) * 1000.0

                meta = {
                    "index": idx,
                    "pts_ms": int(pts_ms),
                    "fps": float(fps),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "source": self.source_uri,
                }
                yield meta, frame

                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
                idx += 1
        finally:
            cap.release()


def _normalize_source(uri: str) -> str:
    """
    Accepts "file:///abs/path.mp4" or "/abs/path.mp4" and returns a cv2-friendly path.
    (RTSP는 추후 여기에서 'rtsp://' 분기로 확장)
    """
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri
